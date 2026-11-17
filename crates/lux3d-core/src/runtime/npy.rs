//! Minimal NumPy `.npy` reader for the TripoSG VAE surface entry point.
//!
//! The reference script (`scripts/inference_vae.py:12-23`) loads its input
//! with `np.load(path, allow_pickle=True).tolist()` and then indexes the
//! result with `data["surface_points"]` / `data["surface_normals"]`.
//!
//! `tolist()` is an `ndarray` method: an `.npz` archive loads as an
//! `NpzFile`, which has no `tolist()`, so the accepted container is a `.npy`
//! holding a pickled dict, written with
//! `np.save(path, {"surface_points": ..., "surface_normals": ...})`. The
//! shipped demo file (`assets/example_data_point/surface_point_demo.npy`)
//! stores `surface_points` as a `trimesh.caching.TrackedArray` (an ndarray
//! subclass) and both arrays as float16.
//!
//! This module therefore implements:
//! - the `.npy` header (magic, version 1/2/3, `descr`/`fortran_order`/`shape`);
//! - a restricted pickle interpreter that only resolves the NumPy array and
//!   dtype reconstructions the reference container uses. `ndarray` subclass
//!   state is rebuilt as a plain array; arbitrary globals/reductions are
//!   rejected instead of executed.

use std::path::Path;

use candle_core::{bail, Result};

/// Keys the reference `load_surface` reads (`inference_vae.py:14-15`).
pub(crate) const SURFACE_POINTS_KEY: &str = "surface_points";
pub(crate) const SURFACE_NORMALS_KEY: &str = "surface_normals";

/// Row-major `[N, 3]` surface positions and normals.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SurfaceArrays {
    pub points: Vec<f32>,
    pub normals: Vec<f32>,
}

impl SurfaceArrays {
    pub fn len(&self) -> usize {
        self.points.len() / 3
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}

/// Read the reference surface container and return row-major `[N, 3]` F32
/// buffers.
pub(crate) fn read_surface_arrays(path: &Path) -> Result<SurfaceArrays> {
    let bytes = std::fs::read(path).map_err(|error| {
        candle_core::Error::Msg(format!(
            "failed to read TripoSG surface input `{}`: {error}",
            path.display()
        ))
    })?;
    let (header, payload_offset) = parse_npy_header(&bytes)?;
    if !matches!(header.dtype, NpyDtype::Object) {
        bail!(
            "TripoSG surface input `{}` holds a plain {:?} array, not the keyed dict \
             the reference `scripts/inference_vae.py` loads; save it with \
             `np.save(path, {{\"surface_points\": [N, 3], \"surface_normals\": [N, 3]}})`",
            path.display(),
            header.dtype
        );
    }
    if !header.shape.is_empty() && header.shape != [1] {
        bail!(
            "TripoSG surface input `{}` holds an object array with shape {:?}; expected \
             the 0-d pickled dict written by `np.save(path, {{...}})`",
            path.display(),
            header.shape
        );
    }
    let payload = &bytes[payload_offset..];
    let root = parse_pickle(payload)?;
    let root = match root {
        Value::Array(PickleArray::Object { shape, items, .. })
            if (shape.is_empty() || shape == [1]) && items.len() == 1 =>
        {
            // `ndarray.tolist()` on the 0-d object array returns the stored dict.
            items.into_iter().next().expect("single item")
        }
        other => bail!(
            "TripoSG surface input `{}` does not hold a single pickled dict object \
             (unexpected root kind {})",
            other.kind(),
            path.display()
        ),
    };
    let Value::Dict(entries) = root else {
        bail!(
            "TripoSG surface input `{}` does not hold the expected keyed dict object",
            path.display()
        );
    };
    let points = dict_array(&entries, SURFACE_POINTS_KEY)?;
    let normals = dict_array(&entries, SURFACE_NORMALS_KEY)?;
    if points.1 != normals.1 {
        bail!(
            "TripoSG surface input `{}`: `{SURFACE_POINTS_KEY}` has {} points but \
             `{SURFACE_NORMALS_KEY}` has {}",
            path.display(),
            points.1,
            normals.1
        );
    }
    Ok(SurfaceArrays {
        points: points.0,
        normals: normals.0,
    })
}

fn dict_array(entries: &[(Value, Value)], key: &str) -> Result<(Vec<f32>, usize)> {
    let value = entries
        .iter()
        .find_map(|(entry_key, value)| match entry_key {
            Value::Str(name) if name == key => Some(value),
            _ => None,
        })
        .ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "TripoSG surface input is missing the `{key}` array"
            ))
        })?;
    let Value::Array(array) = value else {
        bail!("TripoSG surface input `{key}` is not a NumPy array");
    };
    array_to_f32_matrix(array, key)
}

// ---------------------------------------------------------------------------
// `.npy` header
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FloatKind {
    F16,
    F32,
    F64,
}

impl FloatKind {
    fn itemsize(self) -> usize {
        match self {
            Self::F16 => 2,
            Self::F32 => 4,
            Self::F64 => 8,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NpyDtype {
    Object,
    Float { kind: FloatKind, big_endian: bool },
}

impl NpyDtype {
    fn itemsize(self) -> usize {
        match self {
            Self::Object => 8,
            Self::Float { kind, .. } => kind.itemsize(),
        }
    }
}

#[derive(Debug)]
struct NpyHeader {
    dtype: NpyDtype,
    shape: Vec<usize>,
}

fn parse_npy_header(bytes: &[u8]) -> Result<(NpyHeader, usize)> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        bail!("not a NumPy `.npy` file: missing the `\\x93NUMPY` magic");
    }
    let major = bytes[6];
    let minor = bytes[7];
    let (header_len, offset): (usize, usize) = match (major, minor) {
        (1, _) => {
            let len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
            (len, 10)
        }
        (2, _) | (3, _) => {
            if bytes.len() < 12 {
                bail!("truncated NumPy `.npy` header length");
            }
            let len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
            (len, 12)
        }
        _ => bail!("unsupported NumPy `.npy` format version {major}.{minor}"),
    };
    let end = offset
        .checked_add(header_len)
        .filter(|end| *end <= bytes.len())
        .ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "truncated NumPy `.npy` header ({} bytes at offset {offset})",
                header_len
            ))
        })?;
    let header = std::str::from_utf8(&bytes[offset..end]).map_err(|error| {
        candle_core::Error::Msg(format!("NumPy `.npy` header is not UTF-8: {error}"))
    })?;
    let header = header.trim_end_matches(['\n', ' ']);
    let descr = header_string_value(header, "'descr'")?;
    // Object arrays never store row-major data here; validate the key exists
    // and is a bool (a malformed header must not silently parse).
    let _fortran_order = header_bool_value(header, "'fortran_order'")?;
    let shape = header_shape_value(header, "'shape'")?;
    let dtype = parse_dtype(&descr, None)?;
    Ok((NpyHeader { dtype, shape }, end))
}

fn header_key_offset<'a>(header: &'a str, key: &str) -> Result<&'a str> {
    let start = header
        .find(key)
        .ok_or_else(|| candle_core::Error::Msg(format!("`.npy` header is missing `{key}`")))?;
    let rest = &header[start + key.len()..];
    let rest = rest.trim_start();
    let rest = rest
        .strip_prefix(':')
        .ok_or_else(|| candle_core::Error::Msg(format!("`.npy` header `{key}` has no `:`")))?;
    Ok(rest.trim_start())
}

fn header_string_value(header: &str, key: &str) -> Result<String> {
    let rest = header_key_offset(header, key)?;
    let (value, _) = parse_python_string(rest.as_bytes())?;
    Ok(value)
}

fn header_bool_value(header: &str, key: &str) -> Result<bool> {
    let rest = header_key_offset(header, key)?;
    if rest.starts_with("True") {
        Ok(true)
    } else if rest.starts_with("False") {
        Ok(false)
    } else {
        bail!("`.npy` header `{key}` is not a Python bool")
    }
}

fn header_shape_value(header: &str, key: &str) -> Result<Vec<usize>> {
    let rest = header_key_offset(header, key)?;
    parse_python_tuple(rest.as_bytes())
}

fn parse_python_tuple(bytes: &[u8]) -> Result<Vec<usize>> {
    let mut cursor = 0usize;
    skip_spaces(bytes, &mut cursor);
    if bytes.get(cursor) != Some(&b'(') {
        bail!("`.npy` header `shape` is not a tuple");
    }
    cursor += 1;
    let mut shape = Vec::new();
    loop {
        skip_spaces(bytes, &mut cursor);
        match bytes.get(cursor) {
            Some(b')') => return Ok(shape),
            Some(byte) if byte.is_ascii_digit() => {
                let mut value = 0usize;
                while let Some(byte) = bytes.get(cursor) {
                    if !byte.is_ascii_digit() {
                        break;
                    }
                    value = value
                        .checked_mul(10)
                        .and_then(|value| value.checked_add((byte - b'0') as usize))
                        .ok_or_else(|| {
                            candle_core::Error::Msg(
                                "`.npy` header `shape` dimension overflows usize".to_string(),
                            )
                        })?;
                    cursor += 1;
                }
                shape.push(value);
                skip_spaces(bytes, &mut cursor);
                match bytes.get(cursor) {
                    Some(b',') => cursor += 1,
                    Some(b')') => return Ok(shape),
                    _ => bail!("`.npy` header `shape` has a malformed tuple"),
                }
            }
            _ => bail!("`.npy` header `shape` has a malformed tuple"),
        }
    }
}

fn parse_python_string(bytes: &[u8]) -> Result<(String, usize)> {
    let quote = *bytes
        .first()
        .ok_or_else(|| candle_core::Error::Msg("empty Python string".to_string()))?;
    if quote != b'\'' && quote != b'"' {
        bail!("expected a Python string literal");
    }
    let mut text = String::new();
    let mut cursor = 1usize;
    while let Some(&byte) = bytes.get(cursor) {
        cursor += 1;
        match byte {
            byte if byte == quote => return Ok((text, cursor)),
            b'\\' => {
                let escaped = *bytes.get(cursor).ok_or_else(|| {
                    candle_core::Error::Msg("unterminated Python string escape".to_string())
                })?;
                cursor += 1;
                match escaped {
                    b'n' => text.push('\n'),
                    b't' => text.push('\t'),
                    b'r' => text.push('\r'),
                    other => text.push(other as char),
                }
            }
            other => text.push(other as char),
        }
    }
    bail!("unterminated Python string literal")
}

fn parse_dtype(descr: &str, byte_order: Option<char>) -> Result<NpyDtype> {
    let mut chars = descr.chars();
    let first = chars
        .next()
        .ok_or_else(|| candle_core::Error::Msg("empty NumPy dtype descriptor".to_string()))?;
    let (order, rest) = match first {
        '<' | '>' | '|' | '=' => (Some(first), chars.as_str()),
        _ => (byte_order, descr),
    };
    let big_endian = matches!(order, Some('>'));
    match rest {
        "O" | "O8" => Ok(NpyDtype::Object),
        "f2" | "e" => Ok(NpyDtype::Float {
            kind: FloatKind::F16,
            big_endian,
        }),
        "f4" | "f" => Ok(NpyDtype::Float {
            kind: FloatKind::F32,
            big_endian,
        }),
        "f8" | "d" => Ok(NpyDtype::Float {
            kind: FloatKind::F64,
            big_endian,
        }),
        other => {
            bail!("unsupported NumPy dtype `{other}` (the surface reader accepts float16/32/64)")
        }
    }
}

// ---------------------------------------------------------------------------
// Restricted pickle interpreter
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum PickleArray {
    Numeric {
        dtype: NpyDtype,
        shape: Vec<usize>,
        fortran_order: bool,
        bytes: Vec<u8>,
    },
    Object {
        shape: Vec<usize>,
        items: Vec<Value>,
    },
}

#[derive(Debug, Clone)]
enum Value {
    None,
    Bool(bool),
    Int(i64),
    Str(String),
    Bytes(Vec<u8>),
    Tuple(Vec<Value>),
    List(Vec<Value>),
    Dict(Vec<(Value, Value)>),
    Global(String, String),
    DtypeBuilder {
        descr: String,
        byte_order: Option<char>,
    },
    Dtype(NpyDtype),
    ArrayBuilder,
    Array(PickleArray),
}

impl Value {
    /// Compact description for error messages (never dumps array payloads).
    fn kind(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Bool(_) => "bool",
            Self::Int(_) => "int",
            Self::Str(_) => "str",
            Self::Bytes(_) => "bytes",
            Self::Tuple(_) => "tuple",
            Self::List(_) => "list",
            Self::Dict(_) => "dict",
            Self::Global(_, _) => "global",
            Self::DtypeBuilder { .. } => "dtype builder",
            Self::Dtype(_) => "dtype",
            Self::ArrayBuilder => "array builder",
            Self::Array(_) => "array",
        }
    }
}

#[derive(Debug, Clone)]
enum StackItem {
    Value(Value),
    Mark,
}

/// Hard cap on memo slots for the restricted pickle reader. A NumPy surface
/// pickle memoizes only a handful of objects (the dict, one dtype, two
/// ndarray builders), so this is far above what a legitimate surface input
/// needs.
const MAX_PICKLE_MEMO_SLOTS: usize = 1 << 20;

struct PickleParser<'a> {
    data: &'a [u8],
    pos: usize,
    stack: Vec<StackItem>,
    memo: Vec<Value>,
}

fn parse_pickle(data: &[u8]) -> Result<Value> {
    PickleParser {
        data,
        pos: 0,
        stack: Vec::new(),
        memo: Vec::new(),
    }
    .run()
}

impl<'a> PickleParser<'a> {
    fn run(&mut self) -> Result<Value> {
        while let Some(opcode) = self.read_u8()? {
            match opcode {
                b'\x80' => {
                    // PROTO
                    let version = self.read_u8()?.ok_or_else(|| self.eof("PROTO"))?;
                    if version > 5 {
                        bail!("unsupported pickle protocol version {version}");
                    }
                }
                b'\x95' => {
                    // FRAME: skip the (u64) frame length; it is only a hint.
                    self.read_bytes(8)?;
                }
                b'.' => {
                    // STOP
                    if self.stack.len() != 1 {
                        bail!(
                            "pickle STOP with {} values on the stack (expected 1)",
                            self.stack.len()
                        );
                    }
                    match self.stack.pop() {
                        Some(StackItem::Value(value)) => return Ok(value),
                        _ => bail!("pickle STOP reached a marker"),
                    }
                }
                b'(' => self.stack.push(StackItem::Mark),
                b')' => self.push(Value::Tuple(Vec::new())),
                b't' => {
                    // TUPLE from the topmost mark.
                    let values = self.pop_mark()?;
                    self.push(Value::Tuple(values));
                }
                b'\x85' | b'\x86' | b'\x87' => {
                    let count = (opcode - 0x84) as usize;
                    let mut values = Vec::with_capacity(count);
                    for _ in 0..count {
                        values.push(self.pop_value()?);
                    }
                    values.reverse();
                    self.push(Value::Tuple(values));
                }
                b']' => self.push(Value::List(Vec::new())),
                b'}' => self.push(Value::Dict(Vec::new())),
                b'a' => {
                    // APPEND: value to list.
                    let value = self.pop_value()?;
                    match self.pop_value()? {
                        Value::List(mut list) => {
                            list.push(value);
                            self.push(Value::List(list));
                        }
                        other => bail!("pickle APPEND target is not a list: {}", other.kind()),
                    }
                }
                b'e' => {
                    // APPENDS: marked values to list.
                    let values = self.pop_mark()?;
                    match self.pop_value()? {
                        Value::List(mut list) => {
                            list.extend(values);
                            self.push(Value::List(list));
                        }
                        other => bail!("pickle APPENDS target is not a list: {}", other.kind()),
                    }
                }
                b's' => {
                    // SETITEM.
                    let value = self.pop_value()?;
                    let key = self.pop_value()?;
                    match self.pop_value()? {
                        Value::Dict(mut dict) => {
                            dict.push((key, value));
                            self.push(Value::Dict(dict));
                        }
                        other => bail!("pickle SETITEM target is not a dict: {}", other.kind()),
                    }
                }
                b'u' => {
                    // SETITEMS: marked key/value pairs to dict.
                    let values = self.pop_mark()?;
                    if values.len() % 2 != 0 {
                        bail!("pickle SETITEMS has an odd number of stack values");
                    }
                    let mut dict = match self.pop_value()? {
                        Value::Dict(dict) => dict,
                        other => bail!("pickle SETITEMS target is not a dict: {}", other.kind()),
                    };
                    let mut values = values.into_iter();
                    while let (Some(key), Some(value)) = (values.next(), values.next()) {
                        dict.push((key, value));
                    }
                    self.push(Value::Dict(dict));
                }
                b'N' => self.push(Value::None),
                b'\x88' => self.push(Value::Bool(true)),
                b'\x89' => self.push(Value::Bool(false)),
                b'J' => {
                    // BININT (signed i32).
                    let bytes = self.read_bytes(4)?;
                    let value = i32::from_le_bytes(bytes.try_into().expect("4 bytes"));
                    self.push(Value::Int(i64::from(value)));
                }
                b'K' => {
                    let value = self.read_u8()?.ok_or_else(|| self.eof("BININT1"))?;
                    self.push(Value::Int(i64::from(value)));
                }
                b'M' => {
                    let bytes = self.read_bytes(2)?;
                    let value = u16::from_le_bytes(bytes.try_into().expect("2 bytes"));
                    self.push(Value::Int(i64::from(value)));
                }
                b'I' => {
                    // INT: text integer (protocol 0).
                    let line = self.read_line()?;
                    if line == "True" {
                        self.push(Value::Bool(true));
                    } else if line == "False" {
                        self.push(Value::Bool(false));
                    } else {
                        let value = line.trim_end_matches('L').parse::<i64>().map_err(|error| {
                            candle_core::Error::Msg(format!(
                                "pickle INT `{line}` is not an i64: {error}"
                            ))
                        })?;
                        self.push(Value::Int(value));
                    }
                }
                b'L' => {
                    // LONG: text integer.
                    let line = self.read_line()?;
                    let value = line.trim_end_matches('L').parse::<i64>().map_err(|error| {
                        candle_core::Error::Msg(format!(
                            "pickle LONG `{line}` is not an i64: {error}"
                        ))
                    })?;
                    self.push(Value::Int(value));
                }
                b'F' => {
                    bail!("pickle FLOAT values are not supported by the surface reader");
                }
                b'G' => {
                    bail!("pickle BINFLOAT values are not supported by the surface reader");
                }
                b'X' => {
                    // BINUNICODE.
                    let len = self.read_u32()? as usize;
                    let bytes = self.read_bytes(len)?;
                    self.push(Value::Str(decode_utf8(bytes)?));
                }
                b'\x8c' => {
                    // SHORT_BINUNICODE.
                    let len = self
                        .read_u8()?
                        .ok_or_else(|| self.eof("SHORT_BINUNICODE"))?;
                    let bytes = self.read_bytes(len as usize)?;
                    self.push(Value::Str(decode_utf8(bytes)?));
                }
                b'\x8d' => {
                    // BINUNICODE8.
                    let len = self.read_u64()? as usize;
                    let bytes = self.read_bytes(len)?;
                    self.push(Value::Str(decode_utf8(bytes)?));
                }
                b'C' => {
                    // SHORT_BINBYTES.
                    let len = self.read_u8()?.ok_or_else(|| self.eof("SHORT_BINBYTES"))?;
                    let bytes = self.read_bytes(len as usize)?.to_vec();
                    self.push(Value::Bytes(bytes));
                }
                b'B' => {
                    // BINBYTES.
                    let len = self.read_u32()? as usize;
                    let bytes = self.read_bytes(len)?.to_vec();
                    self.push(Value::Bytes(bytes));
                }
                b'\x8e' | b'\x96' => {
                    // BINBYTES8 / BYTEARRAY8.
                    let len = self.read_u64()? as usize;
                    let bytes = self.read_bytes(len)?.to_vec();
                    self.push(Value::Bytes(bytes));
                }
                b'\x94' => {
                    // MEMOIZE.
                    let item = self
                        .stack
                        .last()
                        .ok_or_else(|| {
                            candle_core::Error::Msg("pickle MEMOIZE on empty stack".to_string())
                        })?
                        .clone();
                    match item {
                        StackItem::Value(value) => self.memo.push(value),
                        StackItem::Mark => bail!("pickle MEMOIZE on a marker"),
                    }
                }
                b'q' => {
                    // BINPUT.
                    let index = self.read_u8()?.ok_or_else(|| self.eof("BINPUT"))? as usize;
                    self.memo_put(index)?;
                }
                b'r' => {
                    // LONG_BINPUT.
                    let index = self.read_u32()? as usize;
                    self.memo_put(index)?;
                }
                b'p' => {
                    // PUT: text index.
                    let line = self.read_line()?;
                    let index = line.parse::<usize>().map_err(|error| {
                        candle_core::Error::Msg(format!("pickle PUT `{line}`: {error}"))
                    })?;
                    self.memo_put(index)?;
                }
                b'h' => {
                    // BINGET.
                    let index = self.read_u8()?.ok_or_else(|| self.eof("BINGET"))? as usize;
                    self.memo_get(index)?;
                }
                b'j' => {
                    // LONG_BINGET.
                    let index = self.read_u32()? as usize;
                    self.memo_get(index)?;
                }
                b'g' => {
                    // GET: text index.
                    let line = self.read_line()?;
                    let index = line.parse::<usize>().map_err(|error| {
                        candle_core::Error::Msg(format!("pickle GET `{line}`: {error}"))
                    })?;
                    self.memo_get(index)?;
                }
                b'c' => {
                    // GLOBAL: module and name, each newline terminated.
                    let module = self.read_line()?;
                    let name = self.read_line()?;
                    self.push(Value::Global(module, name));
                }
                b'\x93' => {
                    // STACK_GLOBAL: name then module.
                    let name = self.pop_value()?;
                    let module = self.pop_value()?;
                    match (module, name) {
                        (Value::Str(module), Value::Str(name)) => {
                            self.push(Value::Global(module, name))
                        }
                        (module, name) => {
                            bail!("pickle STACK_GLOBAL expects strings, got {module:?} / {name:?}")
                        }
                    }
                }
                b'R' => {
                    // REDUCE.
                    let args = self.pop_value()?;
                    let callable = self.pop_value()?;
                    let reduced = reduce(&callable, args)?;
                    self.push(reduced);
                }
                b'b' => {
                    // BUILD.
                    let state = self.pop_value()?;
                    let target = self.pop_value()?;
                    let built = build(target, state)?;
                    self.push(built);
                }
                b'0' => {
                    // POP.
                    self.pop_value()?;
                }
                b'1' => {
                    // POP_MARK.
                    self.pop_mark()?;
                }
                b'2' => {
                    // DUP.
                    let value = self.pop_value()?;
                    self.push(value.clone());
                    self.push(value);
                }
                other => {
                    bail!(
                        "unsupported pickle opcode 0x{other:02x} at offset {}",
                        self.pos.saturating_sub(1)
                    );
                }
            }
        }
        bail!("pickle stream ended without STOP")
    }

    fn push(&mut self, value: Value) {
        self.stack.push(StackItem::Value(value));
    }

    fn pop_value(&mut self) -> Result<Value> {
        match self.stack.pop() {
            Some(StackItem::Value(value)) => Ok(value),
            Some(StackItem::Mark) => bail!("pickle stack underflow: found a marker"),
            None => bail!("pickle stack underflow"),
        }
    }

    fn pop_mark(&mut self) -> Result<Vec<Value>> {
        let mut values = Vec::new();
        loop {
            match self.stack.pop() {
                Some(StackItem::Value(value)) => values.push(value),
                Some(StackItem::Mark) => {
                    values.reverse();
                    return Ok(values);
                }
                None => bail!("pickle stack underflow: missing MARK"),
            }
        }
    }

    fn memo_put(&mut self, index: usize) -> Result<()> {
        // A legitimate stream can only define a memo slot by consuming at
        // least one opcode byte, and surface pickles use a handful of slots.
        // Rejecting absurd indices here keeps `resize(index + 1)` bounded: a
        // 6-byte `NONE; LONG_BINPUT 0xffffffff; STOP` pickle must not be able
        // to request a multi-gigabyte memo vector.
        let bound = self.data.len().min(MAX_PICKLE_MEMO_SLOTS);
        if index > bound {
            bail!(
                "malformed pickle: memo index {index} exceeds the allowed bound {bound} \
                 (payload length {})",
                self.data.len()
            );
        }
        let value = self.pop_value()?;
        if self.memo.len() <= index {
            self.memo.resize(index + 1, Value::None);
        }
        self.memo[index] = value.clone();
        self.push(value);
        Ok(())
    }

    fn memo_get(&mut self, index: usize) -> Result<()> {
        let value = self
            .memo
            .get(index)
            .ok_or_else(|| {
                candle_core::Error::Msg(format!("pickle memo index {index} is not defined"))
            })?
            .clone();
        self.push(value);
        Ok(())
    }

    fn read_u8(&mut self) -> Result<Option<u8>> {
        let byte = self.data.get(self.pos).copied();
        if byte.is_some() {
            self.pos += 1;
        }
        Ok(byte)
    }

    fn read_u32(&mut self) -> Result<u32> {
        let bytes = self.read_bytes(4)?;
        Ok(u32::from_le_bytes(bytes.try_into().expect("4 bytes")))
    }

    fn read_u64(&mut self) -> Result<u64> {
        let bytes = self.read_bytes(8)?;
        Ok(u64::from_le_bytes(bytes.try_into().expect("8 bytes")))
    }

    fn read_bytes(&mut self, len: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(len)
            .filter(|end| *end <= self.data.len())
            .ok_or_else(|| {
                candle_core::Error::Msg(format!(
                    "truncated pickle: wanted {len} bytes at offset {}",
                    self.pos
                ))
            })?;
        let bytes = &self.data[self.pos..end];
        self.pos = end;
        Ok(bytes)
    }

    fn read_line(&mut self) -> Result<String> {
        let start = self.pos;
        while let Some(&byte) = self.data.get(self.pos) {
            self.pos += 1;
            if byte == b'\n' {
                let line = &self.data[start..self.pos - 1];
                return Ok(String::from_utf8_lossy(line).into_owned());
            }
        }
        bail!("truncated pickle: unterminated line")
    }

    fn eof(&self, opcode: &str) -> candle_core::Error {
        candle_core::Error::Msg(format!("truncated pickle at {opcode}"))
    }
}

fn decode_utf8(bytes: &[u8]) -> Result<String> {
    std::str::from_utf8(bytes)
        .map(str::to_owned)
        .map_err(|error| candle_core::Error::Msg(format!("pickle string is not UTF-8: {error}")))
}

fn is_ndarray_subtype(module: &str, name: &str) -> bool {
    (matches!(
        module,
        "numpy" | "numpy.core.multiarray" | "numpy._core.multiarray"
    ) && name == "ndarray")
        || (module == "trimesh.caching" && name == "TrackedArray")
}

fn reduce(callable: &Value, args: Value) -> Result<Value> {
    let Value::Global(module, name) = callable else {
        bail!(
            "pickle REDUCE callable is not a supported global: {}",
            callable.kind()
        );
    };
    let args = match args {
        Value::Tuple(args) => args,
        other => bail!("pickle REDUCE arguments are not a tuple: {}", other.kind()),
    };
    match (module.as_str(), name.as_str()) {
        ("numpy.core.multiarray", "_reconstruct") | ("numpy._core.multiarray", "_reconstruct") => {
            // `_reconstruct(subtype, shape, typecode)` returns an empty
            // ndarray subclass; the following BUILD supplies the real state.
            match args.first() {
                Some(Value::Global(subtype_module, subtype_name))
                    if is_ndarray_subtype(subtype_module, subtype_name) =>
                {
                    Ok(Value::ArrayBuilder)
                }
                Some(other) => bail!(
                    "pickle `_reconstruct` subclass is not a supported ndarray subtype: {}",
                    other.kind()
                ),
                None => bail!("pickle `_reconstruct` has no subtype argument"),
            }
        }
        ("numpy", "dtype") => {
            let descr = match args.first() {
                Some(Value::Str(descr)) => descr.clone(),
                Some(Value::Dtype(dtype)) => return Ok(Value::Dtype(*dtype)),
                Some(other) => bail!(
                    "pickle `numpy.dtype` argument is not a string: {}",
                    other.kind()
                ),
                None => bail!("pickle `numpy.dtype` has no descriptor argument"),
            };
            Ok(Value::DtypeBuilder {
                descr,
                byte_order: None,
            })
        }
        other => bail!("pickle REDUCE of unsupported global {}::{name}", other.0),
    }
}

fn build(target: Value, state: Value) -> Result<Value> {
    match target {
        Value::DtypeBuilder { descr, .. } => {
            let byte_order = match &state {
                Value::Tuple(items) => items.get(1).and_then(|value| match value {
                    Value::Str(order) => order.chars().next(),
                    _ => None,
                }),
                _ => None,
            };
            Ok(Value::Dtype(parse_dtype(&descr, byte_order)?))
        }
        Value::Dtype(dtype) => Ok(Value::Dtype(dtype)),
        Value::ArrayBuilder => build_array(state),
        other => bail!("pickle BUILD target is not supported: {}", other.kind()),
    }
}

fn build_array(state: Value) -> Result<Value> {
    let Value::Tuple(state) = state else {
        bail!(
            "pickle ndarray BUILD state is not a tuple: {}",
            state.kind()
        );
    };
    // NumPy >= 1.17 uses `(version, shape, dtype, fortran_order, data)`; the
    // legacy layout is `(shape, dtype, fortran_order, data)`.
    let (shape_value, dtype_value, fortran_value, data_value) = match state.len() {
        5 => (&state[1], &state[2], &state[3], &state[4]),
        4 => (&state[0], &state[1], &state[2], &state[3]),
        other => bail!("pickle ndarray BUILD state has {other} fields (expected 4 or 5)"),
    };
    let shape = match shape_value {
        Value::Tuple(dims) => dims
            .iter()
            .map(|dim| match dim {
                Value::Int(value) if *value >= 0 => Ok(*value as usize),
                other => {
                    bail!(
                        "pickle ndarray shape dimension is not a non-negative int: {}",
                        other.kind()
                    )
                }
            })
            .collect::<Result<Vec<_>>>()?,
        other => bail!(
            "pickle ndarray BUILD shape is not a tuple: {}",
            other.kind()
        ),
    };
    let dtype = match dtype_value {
        Value::Dtype(dtype) => *dtype,
        Value::DtypeBuilder { descr, byte_order } => parse_dtype(descr, *byte_order)?,
        other => bail!(
            "pickle ndarray BUILD dtype is not a dtype: {}",
            other.kind()
        ),
    };
    let fortran_order = match fortran_value {
        Value::Bool(value) => *value,
        other => bail!(
            "pickle ndarray BUILD fortran_order is not a bool: {}",
            other.kind()
        ),
    };
    match dtype {
        NpyDtype::Object => {
            let items = match data_value {
                Value::List(items) => items.clone(),
                Value::Tuple(items) => items.clone(),
                Value::Bytes(bytes) if bytes.is_empty() => Vec::new(),
                other => bail!("pickle object ndarray data is not a list: {}", other.kind()),
            };
            Ok(Value::Array(PickleArray::Object { shape, items }))
        }
        NpyDtype::Float { .. } => {
            let bytes = match data_value {
                Value::Bytes(bytes) => bytes.clone(),
                other => bail!("pickle numeric ndarray data is not bytes: {}", other.kind()),
            };
            let expected = shape
                .iter()
                .try_fold(1usize, |count, dim| count.checked_mul(*dim))
                .and_then(|count| count.checked_mul(dtype.itemsize()));
            if let Some(expected) = expected {
                if bytes.len() != expected {
                    bail!(
                        "pickle ndarray data is {} bytes but shape {shape:?} with {dtype:?} \
                         needs {expected}",
                        bytes.len()
                    );
                }
            }
            Ok(Value::Array(PickleArray::Numeric {
                dtype,
                shape,
                fortran_order,
                bytes,
            }))
        }
    }
}

// ---------------------------------------------------------------------------
// Array conversion
// ---------------------------------------------------------------------------

fn array_to_f32_matrix(array: &PickleArray, key: &str) -> Result<(Vec<f32>, usize)> {
    let PickleArray::Numeric {
        dtype,
        shape,
        fortran_order,
        bytes,
    } = array
    else {
        bail!("TripoSG surface input `{key}` is not a numeric array");
    };
    let NpyDtype::Float { kind, big_endian } = dtype else {
        bail!("TripoSG surface input `{key}` is not a float array");
    };
    if shape.len() != 2 || shape[1] != 3 {
        bail!("TripoSG surface input `{key}` must have shape [N, 3], got {shape:?}");
    }
    let rows = shape[0];
    let cols = shape[1];
    let itemsize = kind.itemsize();
    let elements = rows.checked_mul(cols).ok_or_else(|| {
        candle_core::Error::Msg(format!(
            "TripoSG surface input `{key}` shape {rows}x{cols} overflows"
        ))
    })?;
    let expected = elements.checked_mul(itemsize).ok_or_else(|| {
        candle_core::Error::Msg(format!(
            "TripoSG surface input `{key}` byte length overflows"
        ))
    })?;
    if bytes.len() < expected {
        bail!("TripoSG surface input `{key}` data is shorter than its shape");
    }
    let mut values = Vec::with_capacity(elements);
    for row in 0..rows {
        for col in 0..cols {
            let element = if *fortran_order {
                col * rows + row
            } else {
                row * cols + col
            };
            let offset = element * itemsize;
            let bytes = bytes.get(offset..offset + itemsize).ok_or_else(|| {
                candle_core::Error::Msg(format!(
                    "TripoSG surface input `{key}` data is shorter than its shape"
                ))
            })?;
            values.push(read_float(bytes, *kind, *big_endian));
        }
    }
    Ok((values, rows))
}

fn read_float(bytes: &[u8], kind: FloatKind, big_endian: bool) -> f32 {
    match kind {
        FloatKind::F16 => {
            let bits = if big_endian {
                u16::from_be_bytes([bytes[0], bytes[1]])
            } else {
                u16::from_le_bytes([bytes[0], bytes[1]])
            };
            half_to_f32(bits)
        }
        FloatKind::F32 => {
            let bits = if big_endian {
                u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
            } else {
                u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
            };
            f32::from_bits(bits)
        }
        FloatKind::F64 => {
            let bits = if big_endian {
                u64::from_be_bytes([
                    bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
                ])
            } else {
                u64::from_le_bytes([
                    bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
                ])
            };
            f64::from_bits(bits) as f32
        }
    }
}

/// IEEE 754 binary16 -> binary32 (subnormals included). The surface demo is
/// float16, so this must round-trip the shipped values.
fn half_to_f32(bits: u16) -> f32 {
    let sign = ((bits as u32) & 0x8000) << 16;
    let exponent = ((bits as u32) >> 10) & 0x1f;
    let mantissa = (bits as u32) & 0x03ff;
    let out = if exponent == 0 {
        if mantissa == 0 {
            sign
        } else {
            let mut exponent = -14i32;
            let mut mantissa = mantissa;
            while mantissa & 0x0400 == 0 {
                mantissa <<= 1;
                exponent -= 1;
            }
            mantissa &= 0x03ff;
            sign | (((exponent + 127) as u32) << 23) | (mantissa << 13)
        }
    } else if exponent == 0x1f {
        sign | 0x7f80_0000 | (mantissa << 13)
    } else {
        sign | (((exponent as i32 - 15 + 127) as u32) << 23) | (mantissa << 13)
    };
    f32::from_bits(out)
}

fn skip_spaces(bytes: &[u8], cursor: &mut usize) {
    while bytes.get(*cursor) == Some(&b' ') {
        *cursor += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn test_dir(tag: &str) -> std::path::PathBuf {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "lux3d-npy-{tag}-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).expect("temp dir");
        path
    }

    /// Minimal pickle writer reproducing NumPy's protocol-3 ndarray stream.
    /// The hermetic fixtures are written by this helper; the optional test
    /// below parses a file produced by real NumPy.
    struct PickleWriter {
        out: Vec<u8>,
        memo_next: u8,
    }

    impl PickleWriter {
        fn new() -> Self {
            let mut out = Vec::new();
            out.extend_from_slice(&[b'\x80', 3]); // PROTO 3
            Self { out, memo_next: 0 }
        }

        fn byte(&mut self, byte: u8) {
            self.out.push(byte);
        }

        fn global_memo(&mut self, module: &str, name: &str) {
            self.byte(b'c');
            self.out.extend_from_slice(module.as_bytes());
            self.byte(b'\n');
            self.out.extend_from_slice(name.as_bytes());
            self.byte(b'\n');
            self.binput();
        }

        fn binput(&mut self) -> u8 {
            let index = self.memo_next;
            self.memo_next += 1;
            self.byte(b'q'); // BINPUT
            self.byte(index);
            index
        }

        fn binget(&mut self, index: u8) {
            self.byte(b'h');
            self.byte(index);
        }

        fn binint1(&mut self, value: u8) {
            self.byte(b'K');
            self.byte(value);
        }

        fn binint(&mut self, value: i32) {
            self.byte(b'J');
            self.out.extend_from_slice(&value.to_le_bytes());
        }

        fn newtrue(&mut self) {
            self.byte(b'\x88');
        }

        fn newfalse(&mut self) {
            self.byte(b'\x89');
        }

        fn none(&mut self) {
            self.byte(b'N');
        }

        fn mark(&mut self) {
            self.byte(b'(');
        }

        fn empty_tuple(&mut self) {
            self.byte(b')');
        }

        fn empty_list(&mut self) {
            self.byte(b']');
        }

        fn empty_dict(&mut self) {
            self.byte(b'}');
        }

        fn tuple(&mut self) {
            self.byte(b't');
        }

        fn tuple1(&mut self) {
            self.byte(b'\x85');
        }

        fn tuple2(&mut self) {
            self.byte(b'\x86');
        }

        fn tuple3(&mut self) {
            self.byte(b'\x87');
        }

        fn binunicode(&mut self, text: &str) {
            self.byte(b'X');
            self.out
                .extend_from_slice(&(text.len() as u32).to_le_bytes());
            self.out.extend_from_slice(text.as_bytes());
        }

        fn short_binbytes(&mut self, bytes: &[u8]) {
            assert!(bytes.len() < 256, "test fixture uses SHORT_BINBYTES");
            self.byte(b'C');
            self.byte(bytes.len() as u8);
            self.out.extend_from_slice(bytes);
        }

        fn binbytes(&mut self, bytes: &[u8]) {
            self.byte(b'B');
            self.out
                .extend_from_slice(&(bytes.len() as u32).to_le_bytes());
            self.out.extend_from_slice(bytes);
        }

        fn bytes(&mut self, bytes: &[u8]) {
            if bytes.len() < 256 {
                self.short_binbytes(bytes);
            } else {
                self.binbytes(bytes);
            }
        }

        fn reduce(&mut self) {
            self.byte(b'R');
        }

        fn build(&mut self) {
            self.byte(b'b');
        }

        fn append(&mut self) {
            self.byte(b'a');
        }

        fn setitems(&mut self) {
            self.byte(b'u');
        }

        fn stop(&mut self) {
            self.byte(b'.');
        }

        /// Emit `numpy.dtype(descr)` + its BUILD. Returns the memo index of
        /// the pre-BUILD builder (the object NumPy memoizes and later reuses
        /// via `BINGET`).
        fn dtype(&mut self, descr: &str, order: char) -> u8 {
            self.global_memo("numpy", "dtype");
            self.binunicode(descr);
            self.newfalse();
            self.newtrue();
            self.tuple3();
            self.reduce();
            let memo = self.binput();
            self.mark();
            self.binint1(3);
            self.binunicode(&order.to_string());
            self.none();
            self.none();
            self.none();
            self.binint(-1);
            self.binint(-1);
            self.binint1(0);
            self.tuple();
            self.build();
            memo
        }

        /// Emit `ndarray` reconstruction for one float32 array. `dtype_memo`
        /// reuses a previously built dtype; `None` emits a fresh dtype and
        /// returns its memo index so later arrays can reuse it.
        fn array(
            &mut self,
            shape: [u8; 2],
            dtype_memo: Option<u8>,
            fortran: bool,
            data: &[u8],
        ) -> Option<u8> {
            self.array_with_dtype(shape, dtype_memo, "f4", '<', fortran, data)
        }

        fn array_with_dtype(
            &mut self,
            shape: [u8; 2],
            dtype_memo: Option<u8>,
            descr: &str,
            order: char,
            fortran: bool,
            data: &[u8],
        ) -> Option<u8> {
            self.global_memo("numpy.core.multiarray", "_reconstruct");
            self.global_memo("numpy", "ndarray");
            self.binint1(0);
            self.tuple1();
            self.short_binbytes(b"b");
            self.tuple3();
            self.reduce();
            self.mark();
            self.binint1(1);
            self.binint1(shape[0]);
            self.binint1(shape[1]);
            self.tuple2();
            let created = match dtype_memo {
                Some(index) => {
                    self.binget(index);
                    None
                }
                None => Some(self.dtype(descr, order)),
            };
            if fortran {
                self.newtrue();
            } else {
                self.newfalse();
            }
            self.bytes(data);
            self.tuple();
            self.build();
            created
        }

        /// Emit the 0-d object array wrapping the surface dict.
        fn surface_dict(&mut self, points: &[u8], normals: &[u8], rows: u8) {
            self.global_memo("numpy.core.multiarray", "_reconstruct");
            self.global_memo("numpy", "ndarray");
            self.binint1(0);
            self.tuple1();
            self.short_binbytes(b"b");
            self.tuple3();
            self.reduce();
            self.mark();
            self.binint1(1);
            self.empty_tuple();
            self.dtype("O8", '|');
            self.newfalse();
            self.empty_list();
            self.empty_dict();
            self.mark();
            self.binunicode(SURFACE_POINTS_KEY);
            let dtype_memo = self
                .array([rows, 3], None, false, points)
                .expect("fresh dtype");
            self.binunicode(SURFACE_NORMALS_KEY);
            self.array([rows, 3], Some(dtype_memo), false, normals);
            self.setitems();
            self.append();
            self.tuple();
            self.build();
            self.stop();
        }
    }

    fn npy_bytes(payload: &[u8]) -> Vec<u8> {
        let mut header = b"{'descr': '|O', 'fortran_order': False, 'shape': (), }".to_vec();
        while (10 + header.len() + 1) % 64 != 0 {
            header.push(b' ');
        }
        header.push(b'\n');
        let mut out = Vec::new();
        out.extend_from_slice(b"\x93NUMPY");
        out.extend_from_slice(&[1, 0]);
        out.extend_from_slice(&(header.len() as u16).to_le_bytes());
        out.extend_from_slice(&header);
        out.extend_from_slice(payload);
        out
    }

    fn write_fixture(path: &std::path::Path, bytes: &[u8]) {
        std::fs::write(path, bytes).expect("write .npy fixture");
    }

    #[test]
    fn reads_float32_c_order_surface_dict() {
        let dir = test_dir("c-order");
        let path = dir.join("surface.npy");

        let points: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let normals: Vec<f32> = vec![0.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let points_bytes = points
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<u8>>();
        let normals_bytes = normals
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<u8>>();
        let mut writer = PickleWriter::new();
        writer.surface_dict(&points_bytes, &normals_bytes, 2);
        write_fixture(&path, &npy_bytes(&writer.out));

        let surface = read_surface_arrays(&path).expect("read C-order surface");
        assert_eq!(surface.len(), 2);
        assert_eq!(surface.points, points);
        assert_eq!(surface.normals, normals);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn reads_float16_fortran_order_surface_dict() {
        let dir = test_dir("f-order");
        let path = dir.join("surface_f.npy");

        // Column-major numerics: [[1, 2, 3], [4, 5, 6]] stores 1,4,2,5,3,6.
        let points_f = [1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0];
        let normals_f = [0.0f32, 0.0, 0.0, 1.0, 1.0, 0.0];
        let points_bytes = f32_to_f16_bytes(&points_f);
        let normals_bytes = f32_to_f16_bytes(&normals_f);

        let mut writer = PickleWriter::new();
        writer.global_memo("numpy.core.multiarray", "_reconstruct");
        writer.global_memo("numpy", "ndarray");
        writer.binint1(0);
        writer.tuple1();
        writer.short_binbytes(b"b");
        writer.tuple3();
        writer.reduce();
        writer.mark();
        writer.binint1(1);
        writer.empty_tuple();
        writer.dtype("O8", '|');
        writer.newfalse();
        writer.empty_list();
        writer.empty_dict();
        writer.mark();
        writer.binunicode(SURFACE_POINTS_KEY);
        let dtype_memo = writer
            .array_with_dtype([2, 3], None, "f2", '<', true, &points_bytes)
            .expect("fresh float16 dtype");
        writer.binunicode(SURFACE_NORMALS_KEY);
        writer.array_with_dtype([2, 3], Some(dtype_memo), "f2", '<', true, &normals_bytes);
        writer.setitems();
        writer.append();
        writer.tuple();
        writer.build();
        writer.stop();
        write_fixture(&path, &npy_bytes(&writer.out));

        let surface = read_surface_arrays(&path).expect("read F-order surface");
        assert_eq!(surface.points, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(surface.normals, [0.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_plain_numeric_npy() {
        let dir = test_dir("plain");
        let path = dir.join("plain.npy");
        let mut header = b"{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }".to_vec();
        while (10 + header.len() + 1) % 64 != 0 {
            header.push(b' ');
        }
        header.push(b'\n');
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"\x93NUMPY");
        bytes.extend_from_slice(&[1, 0]);
        bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(&[0u8; 24]);
        write_fixture(&path, &bytes);
        let error = read_surface_arrays(&path).expect_err("plain array is not a dict");
        assert!(error.to_string().contains("plain"), "unexpected: {error}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn f32_to_f16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|&value| f32_to_f16(value).to_le_bytes())
            .collect()
    }

    fn f32_to_f16(value: f32) -> u16 {
        let bits = value.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
        let mantissa = ((bits >> 13) & 0x03ff) as u16;
        if exponent <= 0 {
            return sign;
        }
        if exponent >= 0x1f {
            return sign | 0x7c00;
        }
        sign | ((exponent as u16) << 10) | mantissa
    }

    #[test]
    fn rejects_missing_surface_key() {
        let dir = test_dir("missing");
        let path = dir.join("surface.npy");
        let points_bytes = [0u8; 6 * 4];
        let mut writer = PickleWriter::new();
        writer.surface_dict(&points_bytes, &points_bytes, 2);
        // The helper writes both keys; overwrite the normals key name with a
        // same-length name so the pickle stream stays well formed.
        let text = SURFACE_NORMALS_KEY.as_bytes();
        let position = writer
            .out
            .windows(text.len())
            .position(|window| window == text)
            .expect("normals key");
        writer.out[position..position + text.len()].copy_from_slice(b"surface_normalz");
        write_fixture(&path, &npy_bytes(&writer.out));
        let error = read_surface_arrays(&path).expect_err("missing key");
        assert!(
            error.to_string().contains("surface_normals"),
            "unexpected: {error}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_long_binput_memo_index_above_payload_bound() {
        // `NONE; LONG_BINPUT 0xffffffff; STOP` used to resize the memo vector
        // to roughly 4.3 billion `Value` slots on a 7-byte payload.
        let mut payload = vec![b'N', b'r'];
        payload.extend_from_slice(&u32::MAX.to_le_bytes());
        payload.push(b'.');
        let error = parse_pickle(&payload).expect_err("oversized LONG_BINPUT index");
        let message = error.to_string();
        assert!(
            message.contains("memo index 4294967295"),
            "error must name the offending index: {message}"
        );
        assert!(
            message.contains("allowed bound"),
            "error must name the enforced bound: {message}"
        );
        assert!(
            message.contains("payload length"),
            "error must tie the bound to the payload: {message}"
        );
    }

    #[test]
    fn rejects_binput_memo_index_above_payload_bound() {
        // 4-byte stream: NONE, BINPUT 255, STOP.
        let payload = vec![b'N', b'q', 0xff, b'.'];
        let error = parse_pickle(&payload).expect_err("oversized BINPUT index");
        assert!(
            error.to_string().contains("memo index 255"),
            "unexpected: {error}"
        );
    }

    #[test]
    fn rejects_text_put_memo_index_above_payload_bound() {
        let payload = b"Np4294967295\n.".to_vec();
        let error = parse_pickle(&payload).expect_err("oversized PUT index");
        assert!(
            error.to_string().contains("memo index 4294967295"),
            "unexpected: {error}"
        );
    }

    #[test]
    fn half_to_f32_covers_normals_and_subnormals() {
        for value in [0.0f32, 0.5, 1.0, -1.0, 2.5, 65504.0] {
            assert_eq!(half_to_f32(f32_to_f16(value)), value);
        }
        // Smallest positive subnormal is 2^-24.
        assert_eq!(half_to_f32(1), 2.0f32.powi(-24));
        assert!(half_to_f32(0x7c00).is_infinite());
        assert!(half_to_f32(0x7e00).is_nan());
    }

    #[test]
    fn reads_official_surface_demo_when_available() {
        let path = crate::test_support::runtime_root()
            .join("candle-3d-refs")
            .join("TripoSG")
            .join("assets")
            .join("example_data_point")
            .join("surface_point_demo.npy");
        if !path.is_file() {
            eprintln!(
                "skipping official surface demo check: `{}` is not present",
                path.display()
            );
            return;
        }
        let surface = read_surface_arrays(&path).expect("official surface demo");
        assert_eq!(surface.points.len(), 500_000 * 3);
        assert_eq!(surface.normals.len(), 500_000 * 3);
        assert!(surface.points.iter().all(|value| value.is_finite()));
        assert!(surface.normals.iter().all(|value| value.is_finite()));
    }
}
