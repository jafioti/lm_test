#![allow(clippy::type_complexity)]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
use dataflow::{
    dataloader::Dataloader,
    pipeline::*,
};
use dataflow_nlp::{
    tokenization::{Tokenizer, WordpieceTokenizer},
    vocab::{Vocab, WordPieceVocab},
};
use dfdx::{prelude::*, optim::{Sgd, SgdConfig, Momentum}};

#[allow(unused)]
mod bar;
#[allow(unused)]
#[allow(clippy::uninlined_format_args)]
mod indicatif;

use std::mem::transmute;

use crate::bar::ExponentialAverage;

fn main() {
    let mut train_dataset =
        build_dataset::<5, 10>("/home/jafioti/Datasets/openwebtext", 0, 500_000);
    let dev: Cpu = Default::default();
    let embedding: Tensor2D<30527, 50> = dev.sample_uniform();
    let mut model: (TransformerEncoder<50, 2, 100, 2>, Linear<50, 30527>) = dev.build_module();
    model.reset_params();
    let mut sgd: Sgd<(TransformerEncoder<50, 2, 100, 2>, Linear<50, 30527>)> = Sgd::new(SgdConfig {
        lr: 1e-3,
        momentum: Some(Momentum::Nesterov(0.9)),
        weight_decay: None,
    });

    // Tracking
    let total_len = train_dataset.len();
    let bar = bar::train_progress_bar(train_dataset.len() as u64);
    let mut loss_avg: ExponentialAverage<f32> = bar::ExponentialAverage::new();

    test(&embedding, &model, dev.clone());

    for ((input, target), left) in train_dataset.iter_len() {
        let embedded: Tensor3D<10, 5, 50, _> = embedding.clone().gather(dev.tensor(input));
        let embedded: Tensor3D<2, 25, 50, _> = embedded.reshape();
        let output: Tensor3D<2, 25, 30527, _, OwnedTape<Cpu>> = model.forward_mut(embedded.trace());
        let output: Tensor2D<50, 30527, OwnedTape<Cpu>> = output.reshape();

        let target: Tensor<(Const<50>, Const<30527>), _> = one_hot_encode(&unsafe {transmute(target)}, dev.clone());
        // let loss = cross_entropy_with_logits_loss(output, target);
        let loss = mse_loss(output.log_softmax::<Axis<1>>(), target);
        let loss_val: f32 = loss.array();
        loss_avg.update(loss_val);
        bar.set_position((total_len - left) as u64);
        bar.set_message(format!("PPL: {:.2}", loss_avg.value));

        let gradients = loss.backward();
        sgd.update(&mut model, gradients)
            .expect("Oops, there were some unused params");
    }
    drop(bar);
    println!();
    println!();
    println!();

    test(&embedding, &model, dev.clone());
    test(&embedding, &model, dev);
}

fn test(embedding: &Tensor2D<30527, 50>, model: &(TransformerEncoder<50, 2, 100, 2>, Linear<50, 30527>), dev: Cpu) {
    let input = "Hello there, how are you doing";
    let (tokenizer, vocab) = (WordpieceTokenizer::load(), WordPieceVocab::load());
    let tokens = tokenizer.tokenize(input);
    let indexes = vocab.indexes_from_tokens(&tokens).unwrap();

    let indexes: [usize; 5] = indexes[..5].try_into().unwrap();
    let embedded: Tensor2D<5, 50> = embedding.clone().gather(dev.tensor(indexes));
    let output = model.forward(embedded);
    let indexes: Vec<usize> = output.array().iter().map(|dist| dist
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index)
        .unwrap()).collect();
    
    println!("Result: {}", vocab.tokens_from_indexes(&indexes).unwrap().join(" "));
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
                let strings = strings.into_iter()
                    .filter(|s| !s.is_empty() && !s.contains("\\00")).map(|s| s.replace(['\\', '/', '"'], "")).collect();                
                let tokens = tokenizer.batch_tokenize(strings);
                let indexes = vocab.batch_indexes_from_tokens(&tokens).unwrap();

                indexes.into_iter().filter_map(|indexes| {
                    if indexes.len() > SEQ {
                        Some((
                            TryInto::<[usize; SEQ]>::try_into(indexes[..SEQ].to_vec()).unwrap(),
                            TryInto::<[usize; SEQ]>::try_into(indexes[1..(SEQ + 1).min(indexes.len())].to_vec()).unwrap(),
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

fn one_hot_encode<const B: usize, const T: usize>(
    labels: &[usize; B],
    dev: Cpu
) -> Tensor<(Const<B>, Const<T>), f32> {
    let mut data = vec![0.; B * T];
    for (i, l) in labels.iter().enumerate() {
        data[i* *l] = 1.;
    }

    let mut tensor = dev.zeros();
    tensor.copy_from(&data);
    tensor
}