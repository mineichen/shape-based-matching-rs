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
          podmanRun = "${pkgs.podman}/bin/podman run --rm -it "
            + "--network=slirp4netns "
            + "--tmpfs /tmp "
            + "-v ..:/workspace:z "
            + "-e HOME=/root "
            + "${containername}:latest /bin/entrypoint.sh";
          greet = ''
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
          policy = pkgs.writeText "policy.json" ''{"default":[{"type":"insecureAcceptAnything"}]}'';
        in
        {
          devShells.default = pkgs.mkShell ({
            buildInputs = [ rust ] ++ commonBuildInputs;
            shellHook = greet;
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
          packages.isolated-build = pkgs.dockerTools.buildImage {
            name = containername;
            tag = "latest";
            copyToRoot = pkgs.buildEnv {
              name = containername;
              paths = commonBuildInputs ++ [
                rust
                pkgs.bashInteractive
                pkgs.ripgrep
                pkgs.git
                pkgs.opencode
                pkgs.coreutils
                (pkgs.writeScriptBin "entrypoint.sh" ''
                  #!${pkgs.bashInteractive}/bin/bash
                  ${greet}
                  exec ${pkgs.bashInteractive}/bin/bash
                '')
              ];
              pathsToLink = [ "/bin" "/lib" "/include" "/share" ];
            };
            config = {
              Env = pkgs.lib.mapAttrsToList (k: v: "${k}=${v}") envVars ++ [
                "HOME=/root"
                "SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
                "CARGO_TARGET_DIR=target/opencode"

              ];
              Cmd = [ "/bin/entrypoint.sh" ];
              WorkingDir = "/workspace";
            };
          };
          apps.isolated-build = {
            type = "app";
            program = toString (pkgs.writeShellScript containername ''
              ${pkgs.podman}/bin/podman rmi ${containername} || true
              ${pkgs.podman}/bin/podman load \
                --signature-policy ${policy} \
                --input ${inputs.self.packages.${system}.isolated-build}
              ${podmanRun}
            '');
          };
          apps.isolated-nobuild = {
            type = "app";
            program = toString (pkgs.writeShellScript "run-isolated" ''
              set -euo pipefail
              if ! ${pkgs.podman}/bin/podman image exists ${containername}:latest 2>/dev/null; then
                echo "Image ${containername}:latest not found."
                echo "Please build and load it first with:"
                echo "  nix run .#isolated-build"
                exit 1
              fi
              ${podmanRun}
            '');
          };

        };
    };
}
