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
    transformer::builder::TransformerEncoder<EMBED, HEADS, FF, LAYERS, MAX_LEN>,
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
    transformer::TransformerEncoder<EMBED, HEADS, FF, LAYERS, MAX_LEN, E, D>,
>;