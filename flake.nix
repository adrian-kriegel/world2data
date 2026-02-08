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
          libsm
          libice
          libx11
          libxext
          libxrender
          libxi
          libxfixes
          libxrandr
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
            # make GPU go wroooom
            export __NV_PRIME_RENDER_OFFLOAD=1
            export __GLX_VENDOR_LIBRARY_NAME=nvidia

            # required for pip packages to find libstdc++ etc
            export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath runtimeLibs}:$LD_LIBRARY_PATH
            # OpenCV Qt plugin in pip wheels ships xcb only; force xcb in dev shell.
            export QT_QPA_PLATFORM=''${QT_QPA_PLATFORM:-xcb}
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
