{
  description = "optmz - optimize images/videos, preserve metadata";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let pkgs = import nixpkgs { inherit system; };
      in {
        packages.default = pkgs.stdenv.mkDerivation {
          pname = "optmz";
          version = "1.0.0";
          src = ./.;

          nativeBuildInputs = [ pkgs.makeWrapper ];

          propagatedBuildInputs =
            [ pkgs.python3Packages.pillow pkgs.python3Packages.tqdm ];

          installPhase = ''
            mkdir -p $out/bin
            cp optmz.py $out/bin/optmz
            chmod +x $out/bin/optmz
            wrapProgram $out/bin/optmz \
              --prefix PATH : ${
                pkgs.lib.makeBinPath [
                  pkgs.python3
                  pkgs.ffmpeg
                  pkgs.imagemagick
                ]
              }
          '';
        };
      });
}
