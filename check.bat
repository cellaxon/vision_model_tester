@echo off
echo "Starting Rust code formatting and checks..."

echo "1. Auto-fixing code (cargo fix)"
cargo fix --allow-dirty

echo "2. Code formatting (cargo fmt)"
cargo fmt --all

echo "3. Lint checking (cargo clippy)"
cargo clippy --all-targets --all-features -- -D warnings

echo "4. Running tests"
cargo test

echo "5. Build check"
cargo check

echo "Formatting and checks completed!"
