{
  description = "Deterministic Rust and opencv";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = inputs@{ self, nixpkgs, flake-parts, fenix, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      perSystem = { config, self', inputs', pkgs, system, ... }:
        let
          rust = with fenix.packages.${system}; combine [
            stable.toolchain
            targets.wasm32-unknown-unknown.stable.rust-std
          ];
          rustNightly = with fenix.packages.${system}; combine [
            latest.toolchain
          ];
          commonBuildInputs = [
            pkgs.opencv
            pkgs.pkg-config
            pkgs.cmake
            pkgs.clang
            pkgs.stdenv.cc.cc.lib
            pkgs.bashInteractive
          ];
        in
        {
          devShells.default = pkgs.mkShell {
            buildInputs = [ rust ] ++ commonBuildInputs;

            LIBCLANG_PATH = "${pkgs.clang.cc.lib}/lib";
            PKG_CONFIG_PATH = "${pkgs.opencv}/lib/pkgconfig";
            LD_LIBRARY_PATH = "${pkgs.clang.cc.lib}/lib:${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.opencv}/lib";

            shellHook = ''
              echo "===================================="
              echo " Welcome to the deterministic dev shell! "
              echo "===================================="
              echo "Rust toolchain:"
              rustc --version
              echo "Cargo version:"
              cargo --version
              echo "OpenCV version:"
              pkg-config --modversion opencv4 2>/dev/null || echo "OpenCV available"
              echo "LIBCLANG_PATH: $LIBCLANG_PATH"
              echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
              echo "===================================="
              echo "Ready to develop! 🦀"
            '';
          };

          packages.miri-test = pkgs.writeShellApplication {
            name = "miri-test";
            runtimeInputs = [ rustNightly ] ++ commonBuildInputs;
            text = ''
              set -e
              export LIBCLANG_PATH="${pkgs.clang.cc.lib}/lib"
              export PKG_CONFIG_PATH="${pkgs.opencv}/lib/pkgconfig"
              export LD_LIBRARY_PATH="${pkgs.clang.cc.lib}/lib:${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.opencv}/lib"
              echo "Running Miri tests with nightly toolchain..."
              cargo miri test simd_utils
            '';
          };
        };
    };
}
