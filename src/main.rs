#![allow(clippy::type_complexity)]

use colored::Colorize;
use dataflow::{dataloader::Dataloader, pipeline::*};
use dataflow_nlp::{
    tokenization::{Tokenizer, WordpieceTokenizer},
    vocab::{Vocab, WordPieceVocab},
};
use dfdx::{
    nn::modules,
    optim::{Adam, AdamConfig},
    prelude::*,
};

mod bar;
pub use bar::*;
mod lr_scheduler;
pub use lr_scheduler::*;
use num::Float;
mod indicatif;
mod position_encoding;

type Model<
    const VOCAB: usize,
    const EMBED: usize,
    const FF: usize,
    const LAYERS: usize,
    const HEADS: usize,
    const MAX_LEN: usize,
> = (
    Embedding<VOCAB, EMBED>,
    position_encoding::builder::LearnedPositionalEmbedding<MAX_LEN, EMBED>,
    TransformerEncoder<EMBED, HEADS, FF, LAYERS>,
    Linear<EMBED, VOCAB>,
);

type BuiltModel<
    const VOCAB: usize,
    const EMBED: usize,
    const FF: usize,
    const LAYERS: usize,
    const HEADS: usize,
    const MAX_LEN: usize,
    E,
    D,
> = (
    modules::Embedding<VOCAB, EMBED, E, D>,
    position_encoding::LearnedPositionalEmbedding<MAX_LEN, EMBED, E, D>,
    modules::TransformerEncoder<EMBED, HEADS, FF, LAYERS, E, D>,
    modules::Linear<EMBED, VOCAB, E, D>,
);

// Training
const BATCH_SIZE: usize = 64;
const BATCH_ACCUM: usize = 10;
const BASE_LR: f32 = 1e-5;
const MAX_LR: f32 = 3e-4;

// Model
const LAYERS: usize = 8;
const SEQ_LEN: usize = 25;
const EMBED_DIM: usize = 512;
const FF_DIM: usize = EMBED_DIM * 2;
const HEADS: usize = 8;

fn main() {
    let mut train_dataset = build_dataset::<SEQ_LEN, BATCH_SIZE>(
        "/home/jafioti/Datasets/openwebtext",
        100_000,
        2_000_000,
    );
    let mut test_dataset =
        build_dataset::<SEQ_LEN, BATCH_SIZE>("/home/jafioti/Datasets/openwebtext", 0, 100_000);
    let dev: Cuda = Default::default();
    let mut model =
        Model::<30527, EMBED_DIM, FF_DIM, LAYERS, HEADS, SEQ_LEN>::build_on_device(&dev);
    // model.load("epoch-0.npz").unwrap();
    let mut opt = Adam::new(
        &model,
        AdamConfig {
            lr: 0.,
            ..Default::default()
        },
    );
    let mut scheduler = OneCycleScheduler::new(BASE_LR, MAX_LR);
    let mut tensorboard = Tensorboard::new("logdir");

    println!(
        "Model Parameters: {}",
        pretty_print_num(model.num_trainable_params())
    );

    generate(&model, &dev, 50);
    for epoch in 0..3 {
        println!("{}", format!("Epoch {}", epoch + 1).bold().cyan());
        train_epoch(
            &mut model,
            &mut train_dataset,
            &mut opt,
            &mut scheduler,
            &dev,
            &mut tensorboard,
        );
        println!("Test PPL: {}", test_epoch(&model, &mut test_dataset, &dev));

        generate(&model, &dev, 50);

        if let Err(e) = model.save(&format!("checkpoints/epoch-{epoch}.npz")) {
            println!("{} {e:?}", "Error Saving Model:".bold().red());
        }
    }
}

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
    opt: &mut O,
    scheduler: &mut OneCycleScheduler<f32>,
    dev: &D,
    tensorboard: &mut Tensorboard,
) where
    D: Device<f32>,
    BuiltModel<VOCAB, EMBED, FF, LAYERS, HEADS, SEQ, f32, D>: Module<
            Tensor<(usize, usize), usize, D, OwnedTape<f32, D>>,
            Output = Tensor<(usize, usize, Const<VOCAB>), f32, D, OwnedTape<f32, D>>,
        > + Module<Tensor<(usize,), usize, D>, Output = Tensor<(usize, Const<VOCAB>), f32, D>>,
    Tensor<(), f32, D, OwnedTape<f32, D>>: AsArray<Array = f32>,
    O: Optimizer<BuiltModel<VOCAB, EMBED, FF, LAYERS, HEADS, SEQ, f32, D>, D, f32>
        + LearningRate<f32>,
{
    let total_len = dataset.len();
    let bar = train_progress_bar(dataset.len() as u64);
    let mut loss_avg = ExponentialAverage::with_beta(0.999);
    // let mut gradients: Option<Gradients<f32, D>> = None;
    let mut gradients = Some(model.alloc_grads());
    let accum_samples = BATCH_ACCUM * BATCH_SIZE;
    for ((input, target), left) in dataset.iter_len() {
        // Setup input
        let (batch_size, seq_len) = (input.len(), input[0].len());
        let flat_vec: Vec<usize> = input.into_iter().flatten().collect();
        let input = match Option::take(&mut gradients) {
            Some(g) => dev
                .tensor_from_vec(flat_vec, (batch_size, seq_len))
                .trace_into(g),
            None => dev
                .tensor_from_vec(flat_vec, (batch_size, seq_len))
                .traced(),
        };

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

        // Update status
        loss_avg.update(loss.array().exp()); // Update with PPL
        bar.set_position((total_len - left) as u64);
        bar.set_message(format!("PPL: {:.2}", loss_avg.value));
        tensorboard.record("ppl", loss_avg.value, BATCH_SIZE);

        // Backprop and optimize
        gradients = Some(loss.backward());

        #[allow(clippy::modulo_one)]
        if tensorboard.iter % accum_samples == 0 {
            if let Some(mut grads) = Option::take(&mut gradients) {
                scheduler.set_progress((total_len - left) as f32 / total_len as f32);
                scheduler.step(opt);
                if let Err(e) = opt.update(model, &grads) {
                    println!("{} {e:?}\n", "Update Error:".bold().red());
                }
                model.zero_grads(&mut grads);
                gradients = Some(grads);
            }
        }

        // Save every 200_000 samples
        if tensorboard.iter % 200_000 == 0 {
            if let Err(e) = model.save(&format!("checkpoints/step_{}.npz", tensorboard.iter)) {
                println!("{} {e:?}\n", "Error Saving Model:".bold().red());
            }

            // Run generation
            generate(model, dev, 50);
        }
    }
    drop(bar);

    println!("Train PPL: {}", loss_avg.value);
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
        let loss = loss.array().exp();
        losses.push(loss);
        loss_avg.update(loss);
        bar.set_position((total_len - left) as u64);
        bar.set_message(format!("PPL: {:.2}", loss_avg.value));
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
    model: &BuiltModel<VOCAB, EMBED, FF, LAYERS, HEADS, MAX_LEN, f32, D>,
    dev: &D,
    num_tokens: usize,
) where
    BuiltModel<VOCAB, EMBED, FF, LAYERS, HEADS, MAX_LEN, f32, D>:
        Module<Tensor<(usize,), usize, D>, Output = Tensor<(usize, Const<VOCAB>), f32, D>>,
    D: Device<f32>,
{
    let input = "Hi, how are you doing today? I'd like you to meet my friend Fred. He's a botonist, but also the first man to ever walk on the surface of".to_lowercase();
    let (tokenizer, vocab) = (
        <WordpieceTokenizer as Tokenizer>::load(),
        <WordPieceVocab as Vocab>::load(),
    );
    let tokens = tokenizer.tokenize(&input);
    let mut indexes = vocab.indexes_from_tokens(&tokens).unwrap();
    let initial_len = indexes.len();

    for _ in 0..num_tokens {
        let window = indexes[indexes.len() - 25..].to_vec();
        let output = model.forward(dev.tensor_from_vec(window, (25,)));
        let index = output.as_vec()[24 * VOCAB..]
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(index, _)| index)
            .unwrap();
        indexes.push(index);
    }
    println!(
        "{} {} {}\n",
        "Generation:".bold(),
        tokenizer.untokenize(vocab.tokens_from_indexes(&indexes[..initial_len]).unwrap()),
        tokenizer
            .untokenize(vocab.tokens_from_indexes(&indexes[initial_len..]).unwrap())
            .bold()
    );
}

fn build_dataset<const SEQ: usize, const BATCH: usize>(
    path: &str,
    min_index: usize,
    max_index: usize,
) -> Dataloader<(Vec<Vec<usize>>, Vec<Vec<usize>>)> {
    let pipeline =
    // Random loader from files
    RandomLoader::from_directory(path).min_index(min_index).max_index(max_index)
        // Filter + tokenize + index
        .node(Stateful::new(
            (WordpieceTokenizer::load(), WordPieceVocab::load()),
            |strings: Vec<String>, (tokenizer, vocab)| {
                // Filter / preprocess
                let strings = strings.into_iter()
                    .filter(|s| !s.is_empty() && !s.contains("\\00"))
                    .map(|s| s.to_lowercase().replace(['\\', '/', '"'], "")).collect();
                // Tokenize
                let tokens = tokenizer.batch_tokenize(strings);
                // Convert to indexes
                let indexes = vocab.batch_indexes_from_tokens(&tokens).unwrap();
                // Convert to training examples
                indexes.into_iter().filter_map(|indexes| {
                    if indexes.len() > SEQ {
                        Some((
                            indexes[..SEQ].to_vec(),
                            indexes[1..(SEQ + 1)].to_vec(),
                        ))
                    } else {
                        None
                    }
                }).collect()
            },
        ))
        // Batch
        .node(Batch::new(BATCH))
        // Unzip inputs and targets
        .map(|indexes: Vec<(Vec<usize>, Vec<usize>)>| {
            indexes.into_iter().unzip()
        });
    Dataloader::new(pipeline).load_block_size(10_000)
}

fn _one_hot_encode<const B: usize, const S: usize, const V: usize, D: Device<f32>>(
    labels: &[[usize; S]; B],
    dev: &D,
) -> Result<Tensor<Rank3<B, S, V>, f32, D>, D::Err> {
    let mut data = vec![0.; B * S * V];
    for (b, batch) in labels.iter().enumerate() {
        for (i, l) in batch.iter().enumerate() {
            data[b * S * V + i * V + *l] = 1.;
        }
    }
    dev.try_tensor_from_vec(data, (Const::<B>, Const::<S>, Const::<V>))
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
