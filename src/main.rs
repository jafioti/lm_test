#![allow(clippy::type_complexity)]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use colored::Colorize;
use dataflow::{
    dataloader::Dataloader,
    pipeline::*,
};
use dataflow_nlp::{
    tokenization::{Tokenizer, WordpieceTokenizer},
    vocab::{Vocab, WordPieceVocab},
};
use dfdx::{prelude::*, optim::{Adam, AdamConfig}};

#[allow(unused)]
mod bar;
pub use bar::*;
#[allow(unused)]
#[allow(clippy::uninlined_format_args)]
mod indicatif;
mod position_encoding;

type Model<const VOCAB: usize, const EMBED: usize, const LAYERS: usize, const HEADS: usize, const SEQ: usize> = (
    Embedding<VOCAB, EMBED>,
    position_encoding::builder::LearnedPositionalEmbedding<SEQ, EMBED>,
    TransformerEncoder<EMBED, HEADS, {EMBED * 2}, LAYERS>,
    Linear<EMBED, VOCAB>,
);

type BuiltModel<const VOCAB: usize, const EMBED: usize, const LAYERS: usize, const HEADS: usize, const SEQ: usize, E, D> = (
    dfdx::nn::modules::Embedding<VOCAB, EMBED, E, D>,
    position_encoding::LearnedPositionalEmbedding<SEQ, EMBED, E, D>,
    dfdx::nn::modules::TransformerEncoder<EMBED, HEADS, {EMBED * 2}, LAYERS, E, D>,
    dfdx::nn::modules::Linear<EMBED, VOCAB, E, D>,
);

// Training
const BATCH_SIZE: usize = 32;
const BATCH_ACCUM: usize = 1;
const LR: f32 = 3e-4;

// Model
const LAYERS: usize = 8;
const SEQ_LEN: usize = 25;
const EMBED_DIM: usize = 512;
const HEADS: usize = 16;

fn main() {
    let mut train_dataset = build_dataset::<SEQ_LEN, BATCH_SIZE>("/home/jafioti/Datasets/openwebtext", 100_000, 500_000);
    let mut test_dataset = build_dataset::<SEQ_LEN, BATCH_SIZE>("/home/jafioti/Datasets/openwebtext", 0, 10_000);
    let dev: Cuda = Default::default();
    let mut model = Model::<30527, EMBED_DIM, LAYERS, HEADS, SEQ_LEN>::build_on_device(&dev);
    // model.load("test_save.npz").unwrap();
    let mut opt: Adam<_, _, _> = Adam::new(&model, AdamConfig {
        lr: LR,
        ..Default::default()
    });
    let mut tensorboard = Tensorboard::new("logdir");

    println!("Model Parameters: {}", pretty_print_num(model.num_trainable_params()));

    generate(&model, &dev, 50);
    for epoch in 0..3 {
        println!("{}", format!("Epoch {}", epoch + 1).bold().cyan());
        train_epoch(&mut model, &mut train_dataset, &mut opt, &dev, &mut tensorboard);
        println!("Test PPL: {}", test_epoch(&model, &mut test_dataset, &dev));

        generate(&model, &dev, 50);

        // if let Err(e) = model.save(&format!("model-{epoch}.npz")) {
        //     println!("Error saving model: {e:?}");
        // }
    }
}

fn train_epoch<const LAYERS: usize, const SEQ: usize, const VOCAB: usize, const EMBED: usize, const HEADS: usize, D: Device<f32>, O>(
    model: &mut BuiltModel<VOCAB, EMBED, LAYERS, HEADS, SEQ, f32, D>, 
    dataset: &mut Dataloader<([[usize; SEQ]; BATCH_SIZE], [[usize; SEQ]; BATCH_SIZE])>, 
    opt: &mut O,
    dev: &D,
    tensorboard: &mut Tensorboard,
) where 
    D: Device<f32> + std::fmt::Debug + dfdx::tensor::TensorFrom<[[usize; SEQ]; BATCH_SIZE], Rank2<BATCH_SIZE, SEQ>, usize>,
    BuiltModel<VOCAB, EMBED, LAYERS, HEADS, SEQ, f32, D>: Module<Tensor<Rank2<BATCH_SIZE, SEQ>, usize, D, OwnedTape<f32, D>>, Output = Tensor<Rank3<BATCH_SIZE, SEQ, VOCAB>, f32, D, OwnedTape<f32, D>>>,
    Tensor<(), f32, D, OwnedTape<f32, D>>: AsArray<Array = f32>,
    O: Optimizer<BuiltModel<VOCAB, EMBED, LAYERS, HEADS, SEQ, f32, D>, D, f32>,
    [(); EMBED * 2]:
{
    let total_len = dataset.len();
    let bar = train_progress_bar(dataset.len() as u64);
    let mut loss_avg = ExponentialAverage::<f32>::with_beta(0.999);
    let mut gradients = Some(model.alloc_grads());
    for ((input, target), left) in dataset.iter_len() {
        let input = match gradients.take() {
            Some(g) => dev.tensor(input).trace_into(g),
            None => dev.tensor(input).traced()
        };
        let output = match model.try_forward(input) {
            Ok(o) => o,
            Err(e) => {
                println!("Forward Error: {e:?}");
                continue;
            }
        };

        let loss = match one_hot_encode(&target, dev)
            .map(|t| cross_entropy_with_logits_loss(output, t)) {
            Ok(l) => l,
            r => {
                println!("Loss Error: {r:?}");
                continue;
            }
        };
        
        let ppl = loss.array().exp();
        loss_avg.update(ppl);
        bar.set_position((total_len - left) as u64);
        bar.set_message(format!("PPL: {:.2}", loss_avg.value));
        tensorboard.record("ppl", loss_avg.value, BATCH_SIZE);

        gradients = Some(loss.backward());

        if left % BATCH_ACCUM == 0 {
            if let Some(mut gradients) = gradients.take() {
                if let Err(e) = opt.update(model, &gradients) {
                    println!("Update error: {e:?}");
                }
                model.zero_grads(&mut gradients);
            }
        }

        // if tensorboard.iter * BATCH_SIZE % 100_000 == 0 {
        //     if let Err(e) = model.save(&format!("step_{}.npz", tensorboard.iter * BATCH_SIZE)) {
        //         println!("Error saving model: {e:?}");
        //     }
        // }
    }
    drop(bar);

    println!("Train PPL: {}", loss_avg.value);
}

fn test_epoch<const LAYERS: usize, const SEQ: usize, const VOCAB: usize, const EMBED: usize, const HEADS: usize, D: Device<f32>>(
    model: &BuiltModel<VOCAB, EMBED, LAYERS, HEADS, SEQ, f32, D>, 
    dataset: &mut Dataloader<([[usize; SEQ]; BATCH_SIZE], [[usize; SEQ]; BATCH_SIZE])>, 
    dev: &D
) -> f32
where 
    D: Device<f32> + dfdx::tensor::TensorFrom<[[usize; SEQ]; BATCH_SIZE], Rank2<BATCH_SIZE, SEQ>, usize>,
    Tensor<(), f32, D, NoneTape>: AsArray<Array = f32>,
    BuiltModel<VOCAB, EMBED, LAYERS, HEADS, SEQ, f32, D>: Module<Tensor<Rank2<BATCH_SIZE, SEQ>, usize, D, NoneTape>, Output = Tensor<Rank3<BATCH_SIZE, SEQ, VOCAB>, f32, D, NoneTape>>,
    [(); EMBED * 2]:
{
    let total_len = dataset.len();
    let bar = test_progress_bar(dataset.len() as u64);
    let mut loss_avg = ExponentialAverage::<f32>::new();
    let mut losses = Vec::with_capacity(total_len);
    for ((input, target), left) in dataset.iter_len() {
        let output = model.forward(dev.tensor(input));

        let loss: Tensor<(), f32, D> = cross_entropy_with_logits_loss(output, one_hot_encode(&target, dev).unwrap());
        let loss = loss.array().exp();
        losses.push(loss);
        loss_avg.update(loss);
        bar.set_position((total_len - left) as u64);
        bar.set_message(format!("PPL: {:.2}", loss_avg.value));
    }
    drop(bar);

    return losses.iter().sum::<f32>() / (losses.len() as f32)
}

fn generate<const LAYERS: usize, const VOCAB: usize, const EMBED: usize, const HEADS: usize, D: Device<f32>>(
    model: &BuiltModel<VOCAB, EMBED, LAYERS, HEADS, 25, f32, D>, 
    dev: &D,
    num_tokens: usize,
) 
where 
    // D: TensorToArray<(dfdx::shapes::Const<25>, dfdx::shapes::Const<VOCAB>), f32, Array = [[f32; VOCAB]; 25]>,
    Tensor<(dfdx::shapes::Const<25>, dfdx::shapes::Const<VOCAB>), f32, D>: AsArray<Array = [[f32; VOCAB]; 25]>,
    BuiltModel<VOCAB, EMBED, LAYERS, HEADS, 25, f32, D>: Module<Tensor<Rank1<25>, usize, D>, Output = Tensor<Rank2<25, VOCAB>, f32, D>>,
    D: Device<f32> + dfdx::tensor::TensorFrom<[usize; 25], Rank1<25>, usize> + TensorFrom<[[f32; 25]; 25], Rank2<25, 25>, f32>,
    [(); EMBED * 2]:
{
    let input = "Hi, how are you doing today? I'd like you to meet my friend Fred. He's a botonist, but also the first man to ever walk on the surface of".to_lowercase();
    let (tokenizer, vocab) = (<WordpieceTokenizer as Tokenizer>::load(), <WordPieceVocab as Vocab>::load());
    let tokens = tokenizer.tokenize(&input);
    let mut indexes = vocab.indexes_from_tokens(&tokens).unwrap();

    for _ in 0..num_tokens {
        let window: [usize; 25] = indexes[indexes.len() - 25..].try_into().unwrap();
        let input: Tensor<_, _, D> = dev.tensor(window);
        let output = model.forward(input);
        let index = output.array().last().unwrap()
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(index, _)| index)
            .unwrap();
        indexes.push(index);
    }
    println!("{} {}", "Generation:".bold(), tokenizer.untokenize(vocab.tokens_from_indexes(&indexes).unwrap()));
}

fn build_dataset<const SEQ: usize, const BATCH: usize>(
    path: &str,
    min_index: usize,
    max_index: usize,
) -> Dataloader<(
    [[usize; SEQ]; BATCH],
    [[usize; SEQ]; BATCH],
)> {
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
                            TryInto::<[usize; SEQ]>::try_into(indexes[..SEQ].to_vec()).unwrap(),
                            TryInto::<[usize; SEQ]>::try_into(indexes[1..(SEQ + 1)].to_vec()).unwrap(),
                        ))
                    } else {
                        None
                    }
                }).collect()
            },
        ))
        // Batch
        .node(ArrayBatch::<BATCH, ([usize; SEQ], [usize; SEQ])>::default())
        // Unzip inputs and targets
        .map(|indexes: [([usize; SEQ], [usize; SEQ]); BATCH]| {
            let (inputs, targets): (Vec<[usize; SEQ]>, Vec<[usize; SEQ]>) = indexes.into_iter().unzip();
            (inputs.try_into().unwrap(), targets.try_into().unwrap())
        });
    Dataloader::new(pipeline).load_block_size(1_000_000)
}

fn one_hot_encode<const B: usize, const S: usize, const V: usize, D: Device<f32>>(
    labels: &[[usize; S]; B],
    dev: &D
) -> Result<Tensor<(Const<B>, Const<S>, Const<V>), f32, D>, D::Err> {
    let mut data = vec![0.; B * S * V];
    for (b, batch) in labels.iter().enumerate() {
        for (i, l) in batch.iter().enumerate() {
            data[b*S*V + i*V + *l] = 1.;
        }
    }
    dev.try_tensor_from_vec(data, (Const::<B>, Const::<S>, Const::<V>))
}