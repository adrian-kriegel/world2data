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

        python = pkgs.python3.withPackages (_ps: [
            numpy
            pytest
        ]);

      in {
        devShell = pkgs.mkShell {
          packages = [ python ];

          shellHook = ''
            # IMPORTANT
            # required for python to find libstdc++ etc.
            export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${pkgs.stdenv.cc.cc.lib}/lib/:${pkgs.zlib}/lib/:${pkgs.glib.dev}/lib/:${pkgs.glib}/lib/

            # So local imports like `python -m world2data` resolve in the repo.
            export PYTHONPATH=$PYTHONPATH:$(pwd)
          '';
        };
      }
    );
}
