#![allow(unused)]

use colored::Colorize;
use dataflow::prelude::*;
use dataflow_nlp::{
    pipelines::RandomLoader,
    tokenization::{Tokenizer, WordpieceTokenizer},
    vocab::{Vocab, WordPieceVocab},
};
use dfdx::{
    optim::{Adam, AdamConfig},
    prelude::*,
};
use itertools::Itertools;

use lm_test::utils::{LearningRate, LinearScheduler, OneCycleScheduler, Scheduler};
use num::Float;
use rand::{distributions::WeightedIndex, thread_rng, Rng};
use rand_distr::Distribution;

pub fn openwebtext(
    path: &str,
    min_index: usize,
    max_index: usize,
    max_sequence_length: usize,
    batch_size: usize,
) -> Dataloader<(Vec<Vec<usize>>, Vec<Vec<usize>>)> {
    let pipeline = RandomLoader::from_directory(path)
        .min_index(min_index)
        .max_index(max_index)
        // Filter + tokenize + index
        .chain(
            Stateful::new(
                (WordpieceTokenizer::load(), WordPieceVocab::load()),
                move |strings: Vec<String>, (tokenizer, vocab)| {
                    // Filter / preprocess
                    let strings = strings
                        .into_iter()
                        .filter(|s| !s.is_empty() && !s.contains("\\00"))
                        .map(|s| s.to_lowercase().replace(['\\', '/', '"'], ""))
                        .collect();
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
                            batch
                                .map(|s| {
                                    if reference_length == 0 {
                                        reference_length = s.len().min(max_sequence_length + 1);
                                    }
                                    let mut seq = s[..reference_length].to_vec();
                                    seq.append(&mut vec![
                                        0;
                                        (reference_length)
                                            .checked_sub(seq.len())
                                            .unwrap_or_default()
                                    ]);
                                    (seq[..seq.len() - 1].to_vec(), seq[1..].to_vec())
                                })
                                .collect()
                        })
                        .collect()
                },
            )
            .remaining(move |i| i / batch_size),
        )
        // Re-shuffle
        .chain(Shuffle::default())
        // Unzip inputs and targets
        .map(|indexes: Vec<(Vec<usize>, Vec<usize>)>| indexes.into_iter().unzip());
    Dataloader::new(pipeline).load_block_size(100_000)
}

pub fn simple_openwebtext(
    path: &str,
    min_index: usize,
    max_index: usize,
    sequence_length: usize,
    batch_size: usize,
) -> Dataloader<(Vec<Vec<usize>>, Vec<Vec<usize>>)> {
    let pipeline = OpenWebTextLoader::new(path, sequence_length + 1, min_index, max_index)
        .map(|seq: Vec<usize>| (seq[..seq.len() - 1].to_vec(), seq[1..].to_vec()))
        .chain(Batch::new(batch_size))
        // Unzip inputs and targets
        .map(|indexes: Vec<(Vec<usize>, Vec<usize>)>| indexes.into_iter().unzip());

    Dataloader::new(pipeline).load_block_size(1_000)
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

impl Node<Vec<()>> for OpenWebTextLoader {
    type Output = Vec<Vec<usize>>;

    fn data_remaining(&self, _: usize) -> usize {
        self.total_iter.saturating_sub(self.current_iter)
    }

    fn reset(&mut self) {
        self.current_iter = 0;
    }

    fn process(&mut self, input: Vec<()>) -> Self::Output {
        self.current_iter += input.len();
        let mut rng = thread_rng();
        input
            .iter()
            .map(|_| rng.gen_range(0..self.data.len() - self.seq_len))
            .map(|i| self.data[i..i + self.seq_len].to_vec())
            .collect()
    }
}

pub fn wikitext103(
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
        .chain(|lines: Vec<String>| lines.into_iter().filter(|i| i.len() > 100).collect::<Vec<_>>())
        // Filter + tokenize + index
        .chain(Stateful::new(
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
        .chain(Shuffle::default())
        // Unzip inputs and targets
        .map(|indexes: Vec<(Vec<usize>, Vec<usize>)>| indexes.into_iter().unzip());
    Dataloader::new(pipeline).load_block_size(100_000)
}

pub fn tinystories(
    path: &str,
    min_index: usize,
    max_index: usize,
    max_sequence_length: usize,
    batch_size: usize,
) -> Dataloader<(Vec<Vec<usize>>, Vec<Vec<usize>>)> {
    let pipeline = RandomLoader::new(&[path])
        .max_index(max_index)
        .min_index(min_index)
        .with_delimeter("<|endoftext|>".to_string())
        // Tokenize
        .map(|s: String| s.to_ascii_lowercase())
        .chain(WordpieceTokenizer::load())
        .filter(|line| line.len() >= 10)
        .map(move |mut l: Vec<_>| {
            l.truncate(max_sequence_length);
            l
        })
        .chain(WordPieceVocab::load())
        .map(|mut s: Vec<usize>| {
            s.push(1);
            s
        })
        // Sort
        .chain(|mut seqs: Vec<Vec<usize>>| {
            seqs.sort_by_key(|s| s.len());
            seqs
        })
        .map(|seq: Vec<usize>| (seq[..seq.len() - 1].to_vec(), seq[1..].to_vec()))
        // Batch
        .chain(Batch::new(batch_size))
        .map(|mut batch: Vec<(Vec<usize>, Vec<usize>)>| {
            let max_len = batch.iter().map(|(i, _)| i.len()).max().unwrap();
            for (inp, trg) in &mut batch {
                inp.append(&mut vec![0; max_len - inp.len()]);
                trg.append(&mut vec![0; max_len - trg.len()]);
            }
            batch
        })
        // Shuffle
        .chain(Shuffle::default())
        // Unzip inputs and targets
        .map(|indexes: Vec<(Vec<usize>, Vec<usize>)>| indexes.into_iter().unzip());

    Dataloader::new(pipeline).load_block_size(100_000)
}
