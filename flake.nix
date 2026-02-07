{
  description = "Development environment for world2data.";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
  }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        runtimeLibs = with pkgs; [
          stdenv.cc.cc.lib
          zlib
          glib.dev
          glib
          libxcb
          libglvnd
          mesa
        ];

        python = pkgs.python3.withPackages (ps: with ps; [
          numpy
          openusd
          pytest
        ]);

      in {
        devShell = pkgs.mkShell {
          packages = with pkgs; [
            python

            # for usdview
            openusd
          ];

          shellHook = ''
            # required for pip packages to find libstdc++
            export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath runtimeLibs}:$LD_LIBRARY_PATH
            # expose host GPU driver libs inside nix shell (needed for torch CUDA)
            if [ -d /run/opengl-driver/lib ]; then
              export LD_LIBRARY_PATH=/run/opengl-driver/lib:$LD_LIBRARY_PATH
            fi

            # So local imports like `python -m world2data` resolve in the repo.
            export PYTHONPATH=$PYTHONPATH:$(pwd)
          '';
        };
      }
    );
}
