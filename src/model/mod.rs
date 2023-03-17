pub mod position_encoding;
pub mod reverse_embedding;
pub mod mha;
pub mod transformer;

pub type Model<
    const VOCAB: usize,
    const EMBED: usize,
    const FF: usize,
    const LAYERS: usize,
    const HEADS: usize,
    const MAX_LEN: usize,
> = reverse_embedding::builder::ReverseEmbedding<
    VOCAB,
    EMBED,
    (
        position_encoding::builder::LearnedPositionalEmbedding<MAX_LEN, EMBED>,
        transformer::builder::TransformerEncoder<EMBED, HEADS, FF, LAYERS>,
    ),
>;

pub type BuiltModel<
    const VOCAB: usize,
    const EMBED: usize,
    const FF: usize,
    const LAYERS: usize,
    const HEADS: usize,
    const MAX_LEN: usize,
    E,
    D,
> = reverse_embedding::ReverseEmbedding<
    VOCAB,
    EMBED,
    E,
    D,
    (
        position_encoding::LearnedPositionalEmbedding<MAX_LEN, EMBED, E, D>,
        transformer::TransformerEncoder<EMBED, HEADS, FF, LAYERS, E, D>,
    ),
>;