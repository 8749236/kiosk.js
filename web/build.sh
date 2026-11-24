#!/bin/sh
# Rebuild web/tracemark.wasm from the Rust library.
set -e
cd "$(dirname "$0")/.."
cargo build --lib --release --no-default-features --target wasm32-unknown-unknown
cp target/wasm32-unknown-unknown/release/tracemark.wasm web/tracemark.wasm
ls -l web/tracemark.wasm
