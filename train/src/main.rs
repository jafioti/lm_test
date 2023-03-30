#![allow(clippy::type_complexity)]

use colored::Colorize;
use dataflow::{dataloader::Dataloader, pipeline::*};
use dataflow_nlp::{
    tokenization::{Tokenizer, WordpieceTokenizer},
    vocab::{Vocab, WordPieceVocab},
};
use dfdx::{
    optim::{Adam, AdamConfig},
    prelude::*,
};
use itertools::Itertools;

use lm_test::{
    bar::*,
    lr_scheduler::*,
    model::{BuiltModel, Model},
};
use num::Float;
use rand::{distributions::WeightedIndex, thread_rng, Rng};
use rand_distr::Distribution;

// Training
const BATCH_SIZE: usize = 12;
const BATCH_ACCUM: (usize, usize) = (40, 40);
const MAX_TRAIN_SEQ_LEN: usize = 128;
const LR: (f32, f32) = (6e-4, 6e-4);

// Model
const LAYERS: usize = 8;
const MAX_SEQ_LEN: usize = 512;
const EMBED_DIM: usize = 512;
const FF_DIM: usize = EMBED_DIM * 4;
const HEADS: usize = 8;

fn main() {
    let mut train_dataset = simple_openwebtext(
        "/home/jafioti/Datasets/openwebtext",
        1_500_000,
        5_500_000,
        MAX_TRAIN_SEQ_LEN,
        BATCH_SIZE,
    );
    let mut test_dataset = simple_openwebtext(
        "/home/jafioti/Datasets/openwebtext",
        0,
        10_000,
        MAX_TRAIN_SEQ_LEN,
        BATCH_SIZE,
    );
    // let mut test_dataset = wikitext103(
    //     "/home/jafioti/Datasets/WikiText/wikitext-103/wiki.test.tokens",
    //     0,
    //     usize::MAX,
    //     MAX_TRAIN_SEQ_LEN,
    //     BATCH_SIZE,
    // );
    let dev: Cuda = Default::default();
    let mut model =
        Model::<30528, EMBED_DIM, FF_DIM, LAYERS, HEADS, MAX_SEQ_LEN>::build_on_device(&dev);

    model.load("../checkpoints/step_138240000.npz").unwrap();
    let mut opt = Adam::new(
        &model,
        AdamConfig {
            lr: 0.,
            betas: [0.9, 0.95],
            ..Default::default()
        },
    );
    let mut lr_scheduler = OneCycleScheduler::new(LR.0, LR.1).set_peak(0.2);
    let mut accum_scheduler = LinearScheduler::new(BATCH_ACCUM.0, BATCH_ACCUM.1);
    let mut tensorboard = Tensorboard::new("../logdir");

    println!(
        "Model Parameters: {}",
        pretty_print_num(model.num_trainable_params())
    );

    generate(
        "hi, how are you doing today? i' d like you to meet my friend fred",
        &model,
        &dev,
        50,
        MAX_TRAIN_SEQ_LEN,
        0.5,
        1
    );
    for epoch in 0..3 {
        println!("{}", format!("Epoch {}", epoch + 1).bold().cyan());
        train_epoch(
            &mut model,
            &mut train_dataset,
            &mut test_dataset,
            &mut opt,
            &mut lr_scheduler,
            &mut accum_scheduler,
            &dev,
            &mut tensorboard,
        );
        println!("Val Loss: {}", test_epoch(&model, &mut test_dataset, &dev));

        generate(
            "hi, how are you doing today? i' d like you to meet my friend fred",
            &model,
            &dev,
            50,
            MAX_TRAIN_SEQ_LEN,
            0.5,
            5,
        );

        if let Err(e) = model.save(&format!("../checkpoints/epoch-{epoch}.npz")) {
            println!("{} {e:?}", "Error Saving Model:".bold().red());
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn train_epoch<
    const LAYERS: usize,
    const SEQ: usize,
    const VOCAB: usize,
    const EMBED: usize,
    const FF: usize,
    const HEADS: usize,
    D: Device<f32>,
    O,
>(
    model: &mut BuiltModel<VOCAB, EMBED, FF, LAYERS, HEADS, SEQ, f32, D>,
    dataset: &mut Dataloader<(Vec<Vec<usize>>, Vec<Vec<usize>>)>,
    test_dataset: &mut Dataloader<(Vec<Vec<usize>>, Vec<Vec<usize>>)>,
    opt: &mut O,
    scheduler: &mut OneCycleScheduler<f32>,
    accum_scheduler: &mut LinearScheduler<usize>,
    dev: &D,
    tensorboard: &mut Tensorboard,
) where
    D: Device<f32>,
    BuiltModel<VOCAB, EMBED, FF, LAYERS, HEADS, SEQ, f32, D>: Module<
            Tensor<(usize, usize), usize, D, OwnedTape<f32, D>>,
            Output = Tensor<(usize, usize, Const<VOCAB>), f32, D, OwnedTape<f32, D>>,
        > + Module<Tensor<(usize,), usize, D>, Output = Tensor<(usize, Const<VOCAB>), f32, D>>,
    Tensor<(), f32, D, OwnedTape<f32, D>>: AsArray<Array = f32>,
    Tensor<(), f32, D, NoneTape>: AsArray<Array = f32>,
    O: Optimizer<BuiltModel<VOCAB, EMBED, FF, LAYERS, HEADS, SEQ, f32, D>, D, f32>
        + LearningRate<f32>,
{
    let total_len = dataset.len();
    let bar = train_progress_bar(dataset.len() as u64);
    let mut gradients = Some(model.alloc_grads());
    let mut epoch_iter = 0;
    let mut loss_accum = 0.;
    for ((input, target), left) in dataset.iter_len() {
        epoch_iter += 1;
        accum_scheduler.set_progress((total_len - left) as f32 / total_len as f32);
        // Setup input
        let (batch_size, seq_len) = (input.len(), input[0].len());
        let flat_vec: Vec<usize> = input.into_iter().flatten().collect();
        let input = dev
            .tensor_from_vec(flat_vec, (batch_size, seq_len))
            .trace_into(gradients.take().unwrap_or_else(|| model.alloc_grads()));

        // Run through model
        let output = match model.try_forward(input) {
            Ok(o) => o,
            Err(e) => {
                println!("{} {e:?}\n", "Forward Error:".bold().red());
                continue;
            }
        };

        // Get loss
        let loss = match vec_one_hot_encode(&target, dev)
            .map(|t| try_cross_entropy_with_logits_loss(output, t))
        {
            Ok(Ok(l)) => l,
            r => {
                println!("{} {r:?}\n", "Loss Error:".bold().red());
                continue;
            }
        };

        loss_accum += loss.array();

        // Backprop and optimize
        gradients = Some((loss / accum_scheduler.get() as f32).backward());

        bar.set_position((total_len - left) as u64);
        #[allow(clippy::modulo_one)]
        if epoch_iter % accum_scheduler.get() == 0 {
            // Update status
            bar.set_message(format!(
                "Loss: {:.2}",
                loss_accum / accum_scheduler.get() as f32
            ));
            tensorboard.record(
                "loss",
                loss_accum / accum_scheduler.get() as f32,
                BATCH_SIZE * accum_scheduler.get() * MAX_TRAIN_SEQ_LEN,
            );

            if let Some(mut grads) = Option::take(&mut gradients) {
                scheduler.set_progress((total_len - left) as f32 / total_len as f32);
                scheduler.step(opt);
                if let Err(e) = opt.update(model, &grads) {
                    println!("{} {e:?}\n", "Update Error:".bold().red());
                }
                model.zero_grads(&mut grads);
                gradients = Some(grads);
            }
            loss_accum = 0.;
        }

        // Save every 10_000 steps
        if epoch_iter % 10_000 == 0 {
            if let Err(e) = model.save(&format!("../checkpoints/step_{}.npz", tensorboard.iter)) {
                println!("{} {e:?}\n", "Error Saving Model:".bold().red());
            }

            // Run test
            let val_loss = test_epoch(model, test_dataset, dev);
            println!("Val Loss: {}", format!("{:.2}", val_loss).bold());
            tensorboard.record("val_loss", val_loss, 0);

            // Run generation
            generate(
                "hi, how are you doing today? i' d like you to meet my friend fred",
                model,
                dev,
                50,
                MAX_TRAIN_SEQ_LEN,
                0.5,
                3,
            );
        }
    }
    drop(bar);
}

fn test_epoch<
    const LAYERS: usize,
    const SEQ: usize,
    const VOCAB: usize,
    const EMBED: usize,
    const FF: usize,
    const HEADS: usize,
    D: Device<f32>,
>(
    model: &BuiltModel<VOCAB, EMBED, FF, LAYERS, HEADS, SEQ, f32, D>,
    dataset: &mut Dataloader<(Vec<Vec<usize>>, Vec<Vec<usize>>)>,
    dev: &D,
) -> f32
where
    D: Device<f32>,
    Tensor<(), f32, D, NoneTape>: AsArray<Array = f32>,
    BuiltModel<VOCAB, EMBED, FF, LAYERS, HEADS, SEQ, f32, D>: Module<
        Tensor<(usize, usize), usize, D, NoneTape>,
        Output = Tensor<(usize, usize, Const<VOCAB>), f32, D, NoneTape>,
    >,
{
    let total_len = dataset.len();
    let bar = test_progress_bar(dataset.len() as u64);
    let mut loss_avg = ExponentialAverage::<f32>::new();
    let mut losses = Vec::with_capacity(total_len);
    for ((input, target), left) in dataset.iter_len() {
        let (batch_size, seq_len) = (input.len(), input[0].len());
        let flat_vec: Vec<usize> = input.into_iter().flatten().collect();
        let output = match model.try_forward(dev.tensor_from_vec(flat_vec, (batch_size, seq_len))) {
            Ok(o) => o,
            Err(e) => {
                println!("Forward Error: {e:?}");
                continue;
            }
        };

        let loss = match try_cross_entropy_with_logits_loss(
            output,
            vec_one_hot_encode(&target, dev).unwrap(),
        ) {
            Ok(l) => l,
            Err(e) => {
                println!("Loss Error: {e:?}");
                continue;
            }
        };
        let loss = loss.array();
        losses.push(loss);
        loss_avg.update(loss);
        bar.set_position((total_len - left) as u64);
        bar.set_message(format!("Loss: {:.2}", loss_avg.value));
    }
    drop(bar);

    return losses.iter().sum::<f32>() / (losses.len() as f32);
}

fn generate<
    const LAYERS: usize,
    const VOCAB: usize,
    const EMBED: usize,
    const FF: usize,
    const HEADS: usize,
    const MAX_LEN: usize,
    D: Device<f32>,
>(
    input: &str,
    model: &BuiltModel<VOCAB, EMBED, FF, LAYERS, HEADS, MAX_LEN, f32, D>,
    dev: &D,
    num_tokens: usize,
    window_size: usize,
    temperature: f32,
    generations: u8,
)
where
    BuiltModel<VOCAB, EMBED, FF, LAYERS, HEADS, MAX_LEN, f32, D>:
        Module<Tensor<(usize,), usize, D>, Output = Tensor<(usize, Const<VOCAB>), f32, D>>,
    D: Device<f32>,
{
    let (tokenizer, vocab) = (
        <WordpieceTokenizer as Tokenizer>::load(),
        <WordPieceVocab as Vocab>::load(),
    );
    let tokens = tokenizer.tokenize(&input.to_lowercase());
    for i in 0..generations {
        let mut indexes = vocab.indexes_from_tokens(&tokens).unwrap();
        let initial_len = indexes.len();
        let mut rng = thread_rng();

        for _ in 0..num_tokens {
            let output = model.forward(dev.tensor_from_vec(
                indexes[indexes.len().checked_sub(window_size).unwrap_or_default()..].to_vec(),
                (indexes.len().min(window_size),),
            ));
            let mut distr: Vec<f32> =
                output.as_vec()[(indexes.len() - 1).min(window_size - 1) * VOCAB..].to_vec();
            softmax(&mut distr, temperature);
            indexes.push(WeightedIndex::new(&distr).unwrap().sample(&mut rng));
        }

        println!(
            "{} {} {}\n",
            format!("Generation {}:", i + 1).bold(),
            tokenizer.untokenize(vocab.tokens_from_indexes(&indexes[..initial_len]).unwrap()),
            tokenizer
                .untokenize(vocab.tokens_from_indexes(&indexes[initial_len..]).unwrap())
                .bold(),
        );
    }
}

fn softmax(distr: &mut [f32], temperature: f32) {
    let sum: f32 = distr.iter().map(|i| (i / temperature).exp()).sum();
    for i in distr {
        *i = (*i / temperature).exp() / sum;
    }
}

#[allow(unused)]
fn openwebtext(
    path: &str,
    min_index: usize,
    max_index: usize,
    max_sequence_length: usize,
    batch_size: usize,
) -> Dataloader<(Vec<Vec<usize>>, Vec<Vec<usize>>)> {
    let pipeline =
    // Random loader from files
    // OpenWebTextLoader::new(path, max_sequence_length, min_index, max_index)
    RandomLoader::from_directory(path).min_index(min_index).max_index(max_index)
        // Filter + tokenize + index
        .node(Stateful::new(
            (WordpieceTokenizer::load(), WordPieceVocab::load()),
            move |strings: Vec<String>, (tokenizer, vocab)| {
                // Filter / preprocess
                let strings = strings.into_iter()
                    .filter(|s| !s.is_empty() && !s.contains("\\00"))
                    .map(|s| s.to_lowercase().replace(['\\', '/', '"'], "")).collect();
                // Tokenize
                let tokens = tokenizer.batch_tokenize(strings);
                // Convert to indexes
                let indexes = vocab.batch_indexes_from_tokens(&tokens).unwrap();

                indexes
                    .into_iter()
                    // Filter out sequences shorter than max_sequence_length
                    .filter(|i| i.len() != max_sequence_length)
                    // Sort
                    .sorted_by_key(|a| a.len())
                    // Batch
                    .chunks(batch_size)
                    .into_iter()
                    .map(|batch| {
                        // Limit the length of the vectors and pad out ones that need it
                        let mut reference_length = 0;
                        batch.map(|s| {
                            if reference_length == 0 {
                                reference_length = s.len().min(max_sequence_length + 1);
                            }
                            let mut seq = s[..reference_length].to_vec();
                            seq.append(&mut vec![0; (reference_length).checked_sub(seq.len()).unwrap_or_default()]);
                            (
                                seq[..seq.len() - 1].to_vec(),
                                seq[1..].to_vec()
                            )
                        }).collect()
                    }).collect()
            },
        ).remaining(move |i| i / batch_size))
        // Re-shuffle
        .node(Shuffle::default())
        // Unzip inputs and targets
        .map(|indexes: Vec<(Vec<usize>, Vec<usize>)>| indexes.into_iter().unzip());
    Dataloader::new(pipeline).load_block_size(100_000)
}

fn simple_openwebtext(
    path: &str,
    min_index: usize,
    max_index: usize,
    sequence_length: usize,
    batch_size: usize,
) -> Dataloader<(Vec<Vec<usize>>, Vec<Vec<usize>>)> {
    let pipeline = OpenWebTextLoader::new(path, sequence_length + 1, min_index, max_index)
        .map(|seq: Vec<usize>| (seq[..seq.len() - 1].to_vec(), seq[1..].to_vec()))
        .node(Batch::new(batch_size))
        // Unzip inputs and targets
        .map(|indexes: Vec<(Vec<usize>, Vec<usize>)>| indexes.into_iter().unzip());

    Dataloader::new(pipeline).load_block_size(10_000)
}

struct OpenWebTextLoader {
    data: Vec<usize>,
    seq_len: usize,
    total_iter: usize,
    current_iter: usize,
}

use std::io::{BufRead, BufReader};

impl OpenWebTextLoader {
    fn new(path: &str, seq_len: usize, start_index: usize, max_index: usize) -> Self {
        let mut data = Vec::with_capacity((max_index - start_index) * seq_len);
        let mut curr_index = 0;
        let (tokenizer, vocab) = (WordpieceTokenizer::load(), WordPieceVocab::load());
        for reader in std::fs::read_dir(path).unwrap().map(|r| {
            BufReader::new(std::fs::File::open(r.unwrap().path().to_str().unwrap()).unwrap())
        }) {
            let lines = reader
                .lines()
                .flatten()
                .map(|l| l.to_lowercase().replace(['\\', '/', '"'], ""))
                .collect();
            let tokens = tokenizer.batch_tokenize(lines);

            for seq in tokens {
                if curr_index < (start_index * seq_len) {
                    curr_index += seq.len();
                    continue;
                }
                let mut indexes = vocab.indexes_from_tokens(&seq).unwrap();
                data.append(&mut indexes);

                if data.len() > seq_len * (max_index - start_index) {
                    return Self {
                        data,
                        seq_len,
                        total_iter: max_index - start_index,
                        current_iter: 0,
                    };
                }
            }
        }
        panic!("Not enough data")
    }
}

impl Node for OpenWebTextLoader {
    type Input = Vec<()>;
    type Output = Vec<Vec<usize>>;

    fn data_remaining(&self, _: usize) -> usize {
        self.total_iter.saturating_sub(self.current_iter)
    }

    fn reset(&mut self) {
        self.current_iter = 0;
    }

    fn process(&mut self, input: Self::Input) -> Self::Output {
        self.current_iter += input.len();
        let mut rng = thread_rng();
        input
            .iter()
            .map(|_| rng.gen_range(0..self.data.len() - self.seq_len))
            .map(|i| self.data[i..i + self.seq_len].to_vec())
            .collect()
    }
}

#[allow(unused)]
fn wikitext103(
    path: &str,
    min_index: usize,
    max_index: usize,
    max_sequence_length: usize,
    batch_size: usize,
) -> Dataloader<(Vec<Vec<usize>>, Vec<Vec<usize>>)> {
    let pipeline =
    // Random loader from files
    RandomLoader::new(&[path]).min_index(min_index).max_index(max_index)
        // Filter for length
        .node(|lines: Vec<String>| lines.into_iter().filter(|i| i.len() > 100).collect::<Vec<_>>())
        // Filter + tokenize + index
        .node(Stateful::new(
            (WordpieceTokenizer::load(), WordPieceVocab::load()),
            move |strings: Vec<String>, (tokenizer, vocab)| {
                // Tokenize
                let tokens = tokenizer.batch_tokenize(strings.into_iter()
                    .map(|s| s.to_lowercase().replace(['\\', '/', '"'], "")).collect());
                // Convert to indexes
                let indexes = vocab.batch_indexes_from_tokens(&tokens).unwrap();

                indexes
                    .into_iter()
                    // Filter out sequences shorter than 5
                    .filter(|i| i.len() > 5)
                    // Sort
                    .sorted_by_key(|a| a.len())
                    // Batch
                    .chunks(batch_size)
                    .into_iter()
                    .map(|batch| {
                        // Limit the length of the vectors and pad out ones that need it
                        let mut reference_length = 0;
                        batch.map(|s| {
                            if reference_length == 0 {
                                reference_length = s.len().min(max_sequence_length + 1);
                            }
                            let mut seq = s[..reference_length].to_vec();
                            seq.append(&mut vec![0; (reference_length).checked_sub(seq.len()).unwrap_or_default()]);
                            (
                                seq[..seq.len() - 1].to_vec(),
                                seq[1..].to_vec()
                            )
                        }).collect()
                    }).collect()
            },
        ).remaining(move |i| i / batch_size))
        // Re-shuffle
        .node(Shuffle::default())
        // Unzip inputs and targets
        .map(|indexes: Vec<(Vec<usize>, Vec<usize>)>| indexes.into_iter().unzip());
    Dataloader::new(pipeline).load_block_size(100_000)
}

fn vec_one_hot_encode<const V: usize, E: Dtype + Float, D: Device<E>>(
    labels: &Vec<Vec<usize>>,
    dev: &D,
) -> Result<Tensor<(usize, usize, Const<V>), E, D>, D::Err> {
    let (batch_size, seq_len) = (labels.len(), labels[0].len());
    let mut data = vec![E::zero(); batch_size * seq_len * V];
    for (b, batch) in labels.iter().enumerate() {
        for (i, l) in batch.iter().enumerate() {
            data[b * seq_len * V + i * V + *l] = E::ONE;
        }
    }
    dev.try_tensor_from_vec(data, (batch_size, seq_len, Const::<V>))
}

pub fn try_cross_entropy_with_logits_loss<
    Ax: dfdx::shapes::Axes,
    S,
    E: Dtype,
    D: Device<E>,
    T: Tape<E, D>,
>(
    logits: Tensor<S, E, D, T>,
    target_probs: Tensor<S, E, D>,
) -> Result<Tensor<Rank0, E, D, T>, D::Err>
where
    S: Shape<LastAxis = Ax> + ReduceShape<Ax>,
{
    let last_axis_numel = E::from_usize(<S as HasAxes<Ax>>::size(logits.shape())).unwrap();
    logits
        .try_log_softmax::<Ax>()?
        .try_mul(target_probs)?
        .try_mean()?
        .try_negate()?
        .try_mul(last_axis_numel)
}
