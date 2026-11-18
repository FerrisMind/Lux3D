//! CLIP text conditioning for the TripoSG-scribble pipeline: the reference
//! `CLIPTokenizer` byte-level BPE plus the candle-transformers
//! `ClipTextTransformer` wrapper.
//!
//! Reference semantics (`pipeline_triposg_scribble.py:15-20, 140-147`):
//!
//! - `CLIPTokenizer([prompt], max_length=model_max_length, padding="max_length",
//!   truncation=True)` — the prompt is NFC-normalized, whitespace-collapsed,
//!   lowercased; the BPE is the OpenAI CLIP byte-level BPE with the `</w>`
//!   end-of-word suffix and the CLIP pre-tokenization pattern
//!   `'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+`; the sequence
//!   is `[BOS] + tokens + [EOS]`, truncated to `model_max_length - 2` body
//!   tokens and right-padded with the CLIP pad token.
//! - `CLIPTextModelWithProjection(input_ids).last_hidden_state` — the
//!   PRE-projection `[B, model_max_length, hidden_size]` state (768 for the
//!   shipped scribble checkpoint). The checkpoint's `text_projection` weights
//!   are loaded by neither side of this pipeline.
//!
//! The tokenizer files are read from the checkpoint itself (`vocab.json`,
//! `merges.txt`, `tokenizer_config.json`, `special_tokens_map.json`); the
//! shipped checkpoint has no `tokenizer.json`, so this is a minimal faithful
//! port of the byte-level BPE rather than a `tokenizers`-crate fast tokenizer.
//! Documented deviation: NFC normalization and ftfy's `basic_clean` are not
//! reproduced (Rust std has no Unicode normalization); ASCII prompts — the
//! reference script's surface — tokenize identically, verified against
//! `transformers.CLIPTokenizer` goldens in `tests/scribble_triposg.rs`.

use std::{collections::HashMap, fs, path::Path};

use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_transformers::models::clip::text_model::{
    Activation, ClipTextConfig, ClipTextTransformer,
};

use crate::weights::VarBuilder;

/// OpenAI CLIP `bytes_to_unicode()` plus the inverse decoder.
fn bytes_to_unicode() -> ([char; 256], HashMap<char, u8>) {
    let mut in_table = [false; 256];
    let mut bytes = Vec::with_capacity(256);
    for byte in 33u16..=126 {
        in_table[byte as usize] = true;
        bytes.push(byte as u8);
    }
    for byte in 161u16..=172 {
        in_table[byte as usize] = true;
        bytes.push(byte as u8);
    }
    for byte in 174u16..=255 {
        in_table[byte as usize] = true;
        bytes.push(byte as u8);
    }
    let mut code_points = bytes
        .iter()
        .map(|byte| u32::from(*byte))
        .collect::<Vec<_>>();
    let mut next_code = 256u32;
    for byte in 0u16..=255 {
        if !in_table[byte as usize] {
            bytes.push(byte as u8);
            code_points.push(next_code);
            next_code += 1;
        }
    }
    let mut encoder = ['\0'; 256];
    let mut decoder = HashMap::with_capacity(256);
    for (byte, code_point) in bytes.into_iter().zip(code_points) {
        let ch = char::from_u32(code_point).expect("byte-level code point");
        encoder[usize::from(byte)] = ch;
        decoder.insert(ch, byte);
    }
    (encoder, decoder)
}

/// CLIP byte-level BPE tokenizer.
#[derive(Debug)]
pub struct ClipBpeTokenizer {
    /// Token string (byte-level alphabet plus `</w>` suffixes) -> id.
    encoder: HashMap<String, u32>,
    /// Id -> token string.
    decoder: HashMap<u32, String>,
    /// BPE merge ranks.
    bpe_ranks: HashMap<(String, String), u32>,
    byte_encoder: [char; 256],
    byte_decoder: HashMap<char, u8>,
    /// `tokenizer_config.json` `model_max_length` (77).
    model_max_length: usize,
    bos_token_id: u32,
    eos_token_id: u32,
    pad_token_id: u32,
    unk_token_id: u32,
}

impl ClipBpeTokenizer {
    /// Load `<checkpoint>/tokenizer/{vocab.json,merges.txt,tokenizer_config.json}`
    /// (with the sibling `special_tokens_map.json` when present).
    pub fn from_dir(dir: &Path) -> CandleResult<Self> {
        let config_path = dir.join("tokenizer_config.json");
        let special_map_path = dir.join("special_tokens_map.json");
        let mut config = read_json(&config_path)?;
        if special_map_path.is_file() {
            let special = read_json(&special_map_path)?;
            merge_special_tokens(&mut config, &special);
        }
        Self::from_files(&dir.join("vocab.json"), &dir.join("merges.txt"), &config)
    }

    /// Build from explicit paths plus the already-parsed `tokenizer_config.json`
    /// document (with `special_tokens_map.json` merged in, if any).
    pub fn from_files(
        vocab_path: &Path,
        merges_path: &Path,
        tokenizer_config: &serde_json::Value,
    ) -> CandleResult<Self> {
        let vocab_body = fs::read_to_string(vocab_path).map_err(|error| {
            candle_core::Error::Msg(format!(
                "failed to read CLIP vocab `{}`: {error}",
                vocab_path.display()
            ))
        })?;
        let merges_body = fs::read_to_string(merges_path).map_err(|error| {
            candle_core::Error::Msg(format!(
                "failed to read CLIP merges `{}`: {error}",
                merges_path.display()
            ))
        })?;
        Self::from_json(&vocab_body, &merges_body, tokenizer_config, None)
    }

    /// Build from tokenizer documents embedded in a standalone GGUF. The
    /// optional `special_tokens_map.json` document is merged exactly like
    /// [`Self::from_dir`].
    pub fn from_json(
        vocab_body: &str,
        merges_body: &str,
        tokenizer_config: &serde_json::Value,
        special_tokens_map: Option<&serde_json::Value>,
    ) -> CandleResult<Self> {
        let mut tokenizer_config = tokenizer_config.clone();
        if let Some(special) = special_tokens_map {
            merge_special_tokens(&mut tokenizer_config, special);
        }
        let tokenizer_config = &tokenizer_config;

        let encoder: HashMap<String, u32> = serde_json::from_str(vocab_body).map_err(|error| {
            candle_core::Error::Msg(format!("failed to parse CLIP vocab: {error}"))
        })?;
        let decoder = encoder
            .iter()
            .map(|(token, id)| (*id, token.clone()))
            .collect::<HashMap<_, _>>();

        let mut bpe_ranks = HashMap::new();
        for (rank, line) in merges_body.lines().enumerate() {
            let line = line.trim_end_matches('\r');
            if line.is_empty() || line.starts_with("#version") {
                continue;
            }
            let mut parts = line.split(' ');
            let (Some(first), Some(second), None) = (parts.next(), parts.next(), parts.next())
            else {
                return Err(candle_core::Error::Msg(format!(
                    "invalid CLIP merge line `{line}`"
                )));
            };
            bpe_ranks.insert((first.to_string(), second.to_string()), rank as u32);
        }

        let model_max_length = tokenizer_config
            .get("model_max_length")
            .and_then(|value| value.as_u64())
            .map(|value| value as usize)
            .filter(|value| *value >= 2)
            .unwrap_or(77);
        let (bos_token_id, eos_token_id, pad_token_id, unk_token_id) =
            resolve_special_ids(tokenizer_config, &encoder)?;
        let (byte_encoder, byte_decoder) = bytes_to_unicode();

        Ok(Self {
            encoder,
            decoder,
            bpe_ranks,
            byte_encoder,
            byte_decoder,
            model_max_length,
            bos_token_id,
            eos_token_id,
            pad_token_id,
            unk_token_id,
        })
    }

    pub fn model_max_length(&self) -> usize {
        self.model_max_length
    }

    pub fn bos_token_id(&self) -> u32 {
        self.bos_token_id
    }

    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    pub fn pad_token_id(&self) -> u32 {
        self.pad_token_id
    }

    pub fn unk_token_id(&self) -> u32 {
        self.unk_token_id
    }

    /// Reference tokenizer call: `[prompt]` with `padding="max_length"`,
    /// `max_length=model_max_length`, `truncation=True`.
    pub fn encode(&self, prompt: &str) -> Vec<u32> {
        let text = normalize_text(prompt);
        let mut body = Vec::new();
        for chunk in self.pre_tokenize(&text) {
            match chunk {
                PreToken::Special(id) => body.push(id),
                PreToken::Text(token) => {
                    let byte_encoded = self.byte_encode(&token);
                    for piece in self.bpe(&byte_encoded).split(' ') {
                        if piece.is_empty() {
                            continue;
                        }
                        body.push(
                            self.encoder
                                .get(piece)
                                .copied()
                                .unwrap_or(self.unk_token_id),
                        );
                    }
                }
            }
        }
        body.truncate(self.model_max_length - 2);

        let mut ids = Vec::with_capacity(self.model_max_length);
        ids.push(self.bos_token_id);
        ids.extend(body);
        ids.push(self.eos_token_id);
        ids.resize(self.model_max_length, self.pad_token_id);
        ids
    }

    /// Reverse of [`Self::encode`] over the vocabulary; special tokens are
    /// skipped and the `</w>` suffix becomes a word separator. Used by the
    /// tokenizer round-trip tests.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut output = String::new();
        for id in ids {
            if *id == self.bos_token_id
                || *id == self.eos_token_id
                || *id == self.pad_token_id
                || *id == self.unk_token_id
            {
                continue;
            }
            let Some(token) = self.decoder.get(id) else {
                continue;
            };
            let (body, end_of_word) = match token.strip_suffix("</w>") {
                Some(body) => (body, true),
                None => (token.as_str(), false),
            };
            let bytes = body
                .chars()
                .filter_map(|ch| self.byte_decoder.get(&ch).copied())
                .collect::<Vec<u8>>();
            output.push_str(&String::from_utf8_lossy(&bytes));
            if end_of_word {
                output.push(' ');
            }
        }
        output.trim_end().to_string()
    }

    /// `re.findall(self.pat, text)` equivalent: special tokens, contractions,
    /// letter runs, single numeric characters, and runs of other characters.
    fn pre_tokenize(&self, text: &str) -> Vec<PreToken> {
        let chars = text.chars().collect::<Vec<_>>();
        let mut chunks = Vec::new();
        let mut index = 0usize;
        while index < chars.len() {
            let ch = chars[index];
            if ch.is_whitespace() {
                index += 1;
                continue;
            }
            if let Some((id, consumed)) = self.match_special(&chars[index..]) {
                chunks.push(PreToken::Special(id));
                index += consumed;
                continue;
            }
            if ch == '\'' {
                if let Some(consumed) = match_contraction(&chars[index..]) {
                    chunks.push(PreToken::Text(
                        chars[index..index + consumed].iter().collect(),
                    ));
                    index += consumed;
                    continue;
                }
            }
            if ch.is_alphabetic() {
                let start = index;
                while index < chars.len() && chars[index].is_alphabetic() {
                    index += 1;
                }
                chunks.push(PreToken::Text(chars[start..index].iter().collect()));
                continue;
            }
            if ch.is_numeric() {
                chunks.push(PreToken::Text(ch.to_string()));
                index += 1;
                continue;
            }
            let start = index;
            while index < chars.len() {
                let next = chars[index];
                if next.is_whitespace() || next.is_alphabetic() || next.is_numeric() {
                    break;
                }
                index += 1;
            }
            chunks.push(PreToken::Text(chars[start..index].iter().collect()));
        }
        chunks
    }

    fn match_special(&self, rest: &[char]) -> Option<(u32, usize)> {
        const SPECIALS: [(&str, u32); 2] = [("<|startoftext|>", 49406), ("<|endoftext|>", 49407)];
        for (text, fallback_id) in SPECIALS {
            let special = text.chars().collect::<Vec<_>>();
            if rest.len() >= special.len()
                && rest[..special.len()]
                    .iter()
                    .zip(&special)
                    .all(|(a, b)| a.eq_ignore_ascii_case(b))
            {
                let id = self.encoder.get(text).copied().unwrap_or(fallback_id);
                return Some((id, special.len()));
            }
        }
        None
    }

    fn byte_encode(&self, text: &str) -> String {
        text.as_bytes()
            .iter()
            .map(|byte| self.byte_encoder[usize::from(*byte)])
            .collect()
    }

    /// OpenAI CLIP `bpe(token)`: end-of-word suffix on the last symbol,
    /// lowest-rank merge loop, `" ".join(word)`.
    fn bpe(&self, token: &str) -> String {
        let mut word = token.chars().map(|ch| ch.to_string()).collect::<Vec<_>>();
        if word.is_empty() {
            return String::new();
        }
        let last = word.len() - 1;
        word[last].push_str("</w>");
        if word.len() == 1 {
            return word.join(" ");
        }

        loop {
            let mut best: Option<((String, String), u32)> = None;
            for pair in word.windows(2) {
                let candidate = (pair[0].clone(), pair[1].clone());
                if let Some(rank) = self.bpe_ranks.get(&candidate) {
                    if best.as_ref().is_none_or(|(_, best_rank)| rank < best_rank) {
                        best = Some((candidate, *rank));
                    }
                }
            }
            let Some(((first, second), _)) = best else {
                break;
            };

            let mut merged = Vec::with_capacity(word.len());
            let mut index = 0usize;
            while index < word.len() {
                if index + 1 < word.len() && word[index] == first && word[index + 1] == second {
                    merged.push(format!("{first}{second}"));
                    index += 2;
                    continue;
                }
                merged.push(word[index].clone());
                index += 1;
            }
            word = merged;
            if word.len() == 1 {
                break;
            }
        }
        word.join(" ")
    }
}

#[derive(Debug)]
enum PreToken {
    Special(u32),
    Text(String),
}

/// `CLIPTokenizer` normalizer: whitespace collapse + lowercase (NFC and
/// ftfy's `basic_clean` are not reproduced; see the module docs).
fn normalize_text(text: &str) -> String {
    let lowered = text.to_lowercase();
    let mut output = String::with_capacity(lowered.len());
    let mut pending_space = false;
    for ch in lowered.chars() {
        if ch.is_whitespace() {
            pending_space = !output.is_empty();
        } else {
            if pending_space {
                output.push(' ');
                pending_space = false;
            }
            output.push(ch);
        }
    }
    output
}

fn match_contraction(rest: &[char]) -> Option<usize> {
    const CONTRACTIONS: [&str; 7] = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"];
    for contraction in CONTRACTIONS {
        let chars = contraction.chars().collect::<Vec<_>>();
        if rest.len() >= chars.len()
            && rest[..chars.len()]
                .iter()
                .zip(&chars)
                .all(|(a, b)| a.eq_ignore_ascii_case(b))
        {
            return Some(chars.len());
        }
    }
    None
}

fn read_json(path: &Path) -> CandleResult<serde_json::Value> {
    let body = fs::read_to_string(path).map_err(|error| {
        candle_core::Error::Msg(format!(
            "failed to read CLIP tokenizer config `{}`: {error}",
            path.display()
        ))
    })?;
    serde_json::from_str(&body).map_err(|error| {
        candle_core::Error::Msg(format!(
            "failed to parse CLIP tokenizer config `{}`: {error}",
            path.display()
        ))
    })
}

/// `special_tokens_map.json` values (`{"content": "..."}`) override the plain
/// string entries of `tokenizer_config.json` (the reference `CLIPTokenizer`
/// resolves both into the same special-token table).
fn merge_special_tokens(config: &mut serde_json::Value, special: &serde_json::Value) {
    let Some(config_object) = config.as_object_mut() else {
        return;
    };
    for key in ["bos_token", "eos_token", "pad_token", "unk_token"] {
        let override_value = special.get(key).and_then(|value| {
            value
                .as_str()
                .map(str::to_string)
                .or_else(|| value.get("content")?.as_str().map(str::to_string))
        });
        if let Some(value) = override_value {
            config_object.insert(key.to_string(), serde_json::Value::String(value));
        }
    }
}

fn token_string(value: &serde_json::Value) -> Option<String> {
    value
        .as_str()
        .map(str::to_string)
        .or_else(|| value.get("content")?.as_str().map(str::to_string))
}

fn resolve_special_ids(
    tokenizer_config: &serde_json::Value,
    encoder: &HashMap<String, u32>,
) -> CandleResult<(u32, u32, u32, u32)> {
    let added = tokenizer_config.get("added_tokens_decoder");
    let lookup = |key: &str, fallback: u32| -> u32 {
        if let Some(token) = tokenizer_config.get(key).and_then(token_string) {
            if let Some(id) = encoder.get(&token) {
                return *id;
            }
            if let Some(added) = added.and_then(|value| value.as_object()) {
                for (id, entry) in added {
                    if entry
                        .get("content")
                        .and_then(|value| value.as_str())
                        .is_some_and(|content| content == token)
                    {
                        if let Ok(id) = id.parse::<u32>() {
                            return id;
                        }
                    }
                }
            }
        }
        fallback
    };
    Ok((
        lookup("bos_token", 49406),
        lookup("eos_token", 49407),
        lookup("pad_token", 49407),
        lookup("unk_token", 49407),
    ))
}

/// Candle-transformers CLIP text model loaded from the scribble checkpoint's
/// `text_encoder/model.safetensors`, returning the PRE-projection
/// `last_hidden_state`.
#[derive(Debug)]
pub struct TripoSgClipTextEncoder {
    model: ClipTextTransformer,
    config: ClipTextConfig,
    dtype: DType,
}

impl TripoSgClipTextEncoder {
    /// `vb` must be rooted at the `text_encoder/model.safetensors` shard root
    /// (its keys start with `text_model.`). `config` is that shard's
    /// `config.json`.
    pub fn load(vb: VarBuilder, config: &serde_json::Value, dtype: DType) -> CandleResult<Self> {
        let config = parse_clip_text_config(config)?;
        let model = ClipTextTransformer::new(vb.pp("text_model"), &config)?;
        Ok(Self {
            model,
            config,
            dtype,
        })
    }

    pub fn config(&self) -> &ClipTextConfig {
        &self.config
    }

    /// `CLIPTextModelWithProjection(input_ids).last_hidden_state`:
    /// `[1, input_ids.len(), hidden_size]` in the pipeline dtype.
    pub fn last_hidden_state(&self, input_ids: &[u32], device: &Device) -> CandleResult<Tensor> {
        if input_ids.is_empty() {
            candle_core::bail!("TripoSG CLIP text encoder needs at least one token id");
        }
        if input_ids.len() > self.config.max_position_embeddings {
            candle_core::bail!(
                "TripoSG CLIP text encoder got {} token ids but supports at most {}",
                input_ids.len(),
                self.config.max_position_embeddings
            );
        }
        let ids = Tensor::from_vec(input_ids.to_vec(), (1, input_ids.len()), device)?
            .to_dtype(DType::U32)?;
        // `forward_with_mask(usize::MAX)` is the full-sequence path: the
        // `Module` impl instead returns the pooled EOT row, which is NOT what
        // the scribble pipeline consumes.
        self.model
            .forward_with_mask(&ids, usize::MAX)?
            .to_dtype(self.dtype)
    }

    /// Convenience: tokenize + encode in one call.
    pub fn encode_prompt(
        &self,
        tokenizer: &ClipBpeTokenizer,
        prompt: &str,
        device: &Device,
    ) -> CandleResult<Tensor> {
        if tokenizer.model_max_length() != self.config.max_position_embeddings {
            candle_core::bail!(
                "TripoSG CLIP tokenizer max length {} does not match the text encoder's \
                 max_position_embeddings {}",
                tokenizer.model_max_length(),
                self.config.max_position_embeddings
            );
        }
        self.last_hidden_state(&tokenizer.encode(prompt), device)
    }
}

fn parse_clip_text_config(value: &serde_json::Value) -> CandleResult<ClipTextConfig> {
    let get_usize = |key: &str| -> CandleResult<usize> {
        value
            .get(key)
            .and_then(|field| field.as_u64())
            .map(|field| field as usize)
            .ok_or_else(|| candle_core::Error::Msg(format!("CLIP text config is missing `{key}`")))
    };
    let activation = match value
        .get("hidden_act")
        .and_then(|field| field.as_str())
        .unwrap_or("quick_gelu")
    {
        "quick_gelu" => Activation::QuickGelu,
        other => {
            return Err(candle_core::Error::Msg(format!(
                "unsupported CLIP text activation `{other}` (only quick_gelu is ported)"
            )));
        }
    };
    let config = ClipTextConfig {
        vocab_size: get_usize("vocab_size")?,
        embed_dim: get_usize("hidden_size")?,
        activation,
        intermediate_size: get_usize("intermediate_size")?,
        max_position_embeddings: get_usize("max_position_embeddings")?,
        pad_with: None,
        num_hidden_layers: get_usize("num_hidden_layers")?,
        num_attention_heads: get_usize("num_attention_heads")?,
        projection_dim: get_usize("projection_dim")?,
    };
    if config.embed_dim % config.num_attention_heads != 0 {
        candle_core::bail!(
            "CLIP text hidden_size {} is not divisible by num_attention_heads {}",
            config.embed_dim,
            config.num_attention_heads
        );
    }
    Ok(config)
}

#[cfg(test)]
mod tokenizer_tests {
    use super::*;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_tokenizer_dir(vocab: &str, merges: &str, config: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "lux3d-clip-tokenizer-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).expect("temp dir");
        fs::write(dir.join("vocab.json"), vocab).expect("vocab");
        fs::write(dir.join("merges.txt"), merges).expect("merges");
        fs::write(dir.join("tokenizer_config.json"), config).expect("config");
        dir
    }

    fn tiny_vocab() -> String {
        serde_json::json!({
            "<|startoftext|>": 0,
            "<|endoftext|>": 1,
            "a</w>": 2,
            "b</w>": 3,
            "c</w>": 4,
        })
        .to_string()
    }

    #[test]
    fn byte_level_table_round_trips_every_byte() {
        let (encoder, decoder) = bytes_to_unicode();
        for byte in 0u16..=255 {
            let ch = encoder[usize::from(byte as u8)];
            assert_eq!(decoder.get(&ch), Some(&(byte as u8)));
        }
        assert_eq!(encoder[usize::from(b'a')], 'a');
        assert_eq!(encoder[usize::from(b' ')], '\u{0120}');
    }

    #[test]
    fn encodes_specials_padding_and_lowercase() {
        let dir = temp_tokenizer_dir(
            &tiny_vocab(),
            "#version: 0.2\n",
            r#"{"model_max_length": 5, "bos_token": "<|startoftext|>", "eos_token": "<|endoftext|>", "pad_token": "<|endoftext|>", "unk_token": "<|endoftext|>"}"#,
        );
        let tokenizer = ClipBpeTokenizer::from_dir(&dir).expect("tokenizer");
        assert_eq!(tokenizer.model_max_length(), 5);
        // Whitespace collapse + lowercase: "A  B" -> a, b.
        assert_eq!(tokenizer.encode(" A  B "), vec![0, 2, 3, 1, 1]);
        // Round trip over the constructed vocabulary.
        assert_eq!(tokenizer.decode(&[0, 2, 3, 1, 1]), "a b");
        fs::remove_dir_all(dir).expect("cleanup");
    }

    #[test]
    fn truncates_body_and_pads_to_model_max_length() {
        let dir = temp_tokenizer_dir(
            &tiny_vocab(),
            "#version: 0.2\n",
            r#"{"model_max_length": 4, "bos_token": "<|startoftext|>", "eos_token": "<|endoftext|>", "pad_token": "<|endoftext|>", "unk_token": "<|endoftext|>"}"#,
        );
        let tokenizer = ClipBpeTokenizer::from_dir(&dir).expect("tokenizer");
        // 4 body tokens truncate to model_max_length - 2 = 2.
        let ids = tokenizer.encode("a b a b");
        assert_eq!(ids.len(), 4);
        assert_eq!(ids, vec![0, 2, 3, 1]);
        // Short prompts are right-padded with the pad token id.
        let padded = tokenizer.encode("a");
        assert_eq!(padded, vec![0, 2, 1, 1]);
        fs::remove_dir_all(dir).expect("cleanup");
    }

    #[test]
    fn merges_are_applied_by_rank() {
        // Byte-level "a" is 'a'; build "a</w>" directly via a merge of
        // "a" and "</w>" so the test also covers the merge loop.
        let vocab = serde_json::json!({
            "<|startoftext|>": 0,
            "<|endoftext|>": 1,
            "a": 2,
            "</w>": 3,
            "a</w>": 4,
        })
        .to_string();
        let dir = temp_tokenizer_dir(
            &vocab,
            "#version: 0.2\na </w>\n",
            r#"{"model_max_length": 3, "bos_token": "<|startoftext|>", "eos_token": "<|endoftext|>", "pad_token": "<|endoftext|>", "unk_token": "<|endoftext|>"}"#,
        );
        let tokenizer = ClipBpeTokenizer::from_dir(&dir).expect("tokenizer");
        assert_eq!(tokenizer.encode("a"), vec![0, 4, 1]);
        fs::remove_dir_all(dir).expect("cleanup");
    }
}
