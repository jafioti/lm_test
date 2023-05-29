#![allow(clippy::type_complexity, clippy::too_many_arguments)]

mod data;

use colored::Colorize;
use dataflow::dataloader::Dataloader;
use dataflow_nlp::{
    tokenization::{Tokenizer, WordpieceTokenizer},
    vocab::{Vocab, WordPieceVocab},
};
use dfdx::{
    optim::{Adam, AdamConfig},
    prelude::*,
};

use lm_test::utils::*;
use rand::{distributions::WeightedIndex, thread_rng};
use rand_distr::Distribution;

// Training
const BATCH_SIZE: usize = 4;
const BATCH_ACCUM: (usize, usize) = (1, 1);
const MAX_TRAIN_SEQ_LEN: usize = 124;
const LR: (f64, f64) = (6e-4, 6e-4);

// Model
const LAYERS: usize = 2;
const MAX_SEQ_LEN: usize = 512;
const EMBED_DIM: usize = 512;
const FF_DIM: usize = EMBED_DIM * 4;
const HEADS: usize = 8;
const VOCAB: usize = 30528;

type Model = lm_test::model::Model<VOCAB, EMBED_DIM, FF_DIM, LAYERS, HEADS, MAX_SEQ_LEN>;
type BuiltModel<E, D> = <Model as BuildOnDevice<D, E>>::Built;

fn main() {
    let mut train_dataset = data::tinystories(
        "/Users/jafioti/Downloads/TinyStories-valid.txt",
        0,
        10_000,
        MAX_TRAIN_SEQ_LEN,
        BATCH_SIZE,
    );
    let mut test_dataset = data::tinystories(
        "/Users/jafioti/Downloads/TinyStories-valid.txt",
        0,
        1_000,
        MAX_TRAIN_SEQ_LEN,
        BATCH_SIZE,
    );
    let dev = Cpu::default();
    let mut model = Model::build_on_device(&dev);

    let mut opt = Adam::new(
        &model,
        AdamConfig {
            lr: 0.,
            betas: [0.9, 0.95],
            ..Default::default()
        },
    );
    let mut lr_scheduler = OneCycleScheduler::new(LR.0, LR.1, train_dataset.len()).set_peak(0.2);
    let mut accum_scheduler =
        LinearScheduler::new(BATCH_ACCUM.0, BATCH_ACCUM.1, train_dataset.len());
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
        1,
    );
    for epoch in 0..3 {
        println!("{}", format!("Epoch {}", epoch + 1).bold().cyan());
        println!(
            "Train Loss: {}",
            train_epoch(
                &mut model,
                &mut train_dataset,
                &mut test_dataset,
                &mut opt,
                &mut lr_scheduler,
                &mut accum_scheduler,
                &dev,
                &mut tensorboard,
            )
            .to_string()
            .bold()
            .bright_green()
        );
        println!(
            "Val Loss: {}",
            test_epoch(&model, &mut test_dataset, &dev)
                .to_string()
                .bold()
                .bright_yellow()
        );

        generate(
            "hi, how are you doing today? i' d like you to meet my friend fred",
            &model,
            &dev,
            50,
            MAX_TRAIN_SEQ_LEN,
            0.5,
            5,
        );

        if let Err(e) = model.save(format!("../checkpoints/epoch-{epoch}.npz")) {
            println!("{} {e:?}", "Error Saving Model:".bold().red());
        }
    }
}

fn train_epoch<D: Device<f32>, O, L: Scheduler<f64>, A: Scheduler<usize>>(
    model: &mut BuiltModel<f32, D>,
    dataset: &mut Dataloader<(Vec<Vec<usize>>, Vec<Vec<usize>>)>,
    test_dataset: &mut Dataloader<(Vec<Vec<usize>>, Vec<Vec<usize>>)>,
    opt: &mut O,
    scheduler: &mut L,
    accum_scheduler: &mut A,
    dev: &D,
    tensorboard: &mut Tensorboard,
) -> f32
where
    D: Device<f32>,
    BuiltModel<f32, D>: Module<
            Tensor<(usize, usize), usize, D, OwnedTape<f32, D>>,
            Output = Tensor<(usize, usize, Const<VOCAB>), f32, D, OwnedTape<f32, D>>,
        > + Module<Tensor<(usize,), usize, D>, Output = Tensor<(usize, Const<VOCAB>), f32, D>>,
    Tensor<(), f32, D, OwnedTape<f32, D>>: AsArray<Array = f32>,
    Tensor<(), f32, D, NoneTape>: AsArray<Array = f32>,
    O: Optimizer<BuiltModel<f32, D>, D, f32> + LearningRate,
{
    let total_len = dataset.len();
    let mut loss_ema = ExponentialAverage::<f32>::new();
    let bar = train_progress_bar(dataset.len() as u64);
    let mut gradients = Some(model.alloc_grads());
    let mut loss_accum = 0.;
    for (epoch_iter, ((input, target), left)) in dataset.iter_len().enumerate() {
        // Setup input
        let (batch_size, seq_len) = (input.len(), input[0].len());
        let flat_vec: Vec<usize> = input.into_iter().flatten().collect();
        let input = dev
            .tensor_from_vec(flat_vec, (batch_size, seq_len))
            .traced(gradients.take().unwrap_or_else(|| model.alloc_grads()));

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
        accum_scheduler.step(batch_size);
        scheduler.step(batch_size);
        if epoch_iter % accum_scheduler.get() == 0 {
            // Update status
            loss_ema.update(loss_accum / accum_scheduler.get() as f32);
            bar.set_message(format!("Loss: {:.2}", loss_ema.value));
            tensorboard.record(
                "loss",
                loss_accum / accum_scheduler.get() as f32,
                BATCH_SIZE * accum_scheduler.get() * MAX_TRAIN_SEQ_LEN,
            );

            if let Some(mut grads) = Option::take(&mut gradients) {
                *opt.learning_rate() = scheduler.get();
                if let Err(e) = opt.update(model, &grads) {
                    println!("{} {e:?}\n", "Update Error:".bold().red());
                }
                if let Err(e) = model.try_zero_grads(&mut grads) {
                    println!("{} {e:?}\n", "Zero Grads Error:".bold().red());
                }
                gradients = Some(grads);
            }
            loss_accum = 0.;
        }

        // Save every 10_000 steps
        if epoch_iter % 10_000 == 0 && epoch_iter > 0 {
            if let Err(e) = model.save(format!("../checkpoints/step_{}.npz", tensorboard.iter)) {
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
    loss_ema.value
}

fn test_epoch<D: Device<f32>>(
    model: &BuiltModel<f32, D>,
    dataset: &mut Dataloader<(Vec<Vec<usize>>, Vec<Vec<usize>>)>,
    dev: &D,
) -> f32
where
    D: Device<f32>,
    Tensor<(), f32, D, NoneTape>: AsArray<Array = f32>,
    BuiltModel<f32, D>: Module<
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

fn generate<D: Device<f32>>(
    input: &str,
    model: &BuiltModel<f32, D>,
    dev: &D,
    num_tokens: usize,
    window_size: usize,
    temperature: f32,
    generations: u8,
) where
    BuiltModel<f32, D>:
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
            lm_test::utils::softmax(&mut distr, temperature);
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
