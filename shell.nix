{ pkgs ? import <nixpkgs> { } }:

pkgs.mkShell {
  name = "optmz-dev";

  buildInputs = with pkgs; [
    python3
    python3Packages.tqdm
    python3Packages.pillow
    ffmpeg
    imagemagick
    loupe
    vlc
  ];

  shellHook = ''
    echo "🧠 optmz-dev shell ready with ffmpeg, imagemagick, loupe, and vlc!"
  '';
}
