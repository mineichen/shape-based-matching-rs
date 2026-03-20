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
          envVars = {
            LIBCLANG_PATH = "${pkgs.clang.cc.lib}/lib";
            LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
              pkgs.clang.cc.lib
              pkgs.stdenv.cc.cc.lib
              pkgs.opencv
            ];
          };
          containername = "shape-based-matching-isolated-dev";
        in
        {
          devShells.default = pkgs.mkShell ({
            buildInputs = [ rust ] ++ commonBuildInputs;

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
          } // envVars);

          packages.miri-test = pkgs.writeShellApplication {
            name = "miri-test";
            runtimeInputs = [ rustNightly ] ++ commonBuildInputs;
            text = ''
              set -e
              ${pkgs.lib.concatStringsSep "\n" (pkgs.lib.mapAttrsToList (k: v: "export ${k}=\"${v}\"") envVars)}
              echo "Running Miri tests with nightly toolchain..."
              cargo miri test simd_utils
            '';
          };
        };
    };
}
