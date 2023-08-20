{pkgs, ...}:

with pkgs; [
python310
python310Packages.pip
python310Packages.pytorch-bin
python310Packages.virtualenv
python310Packages.sounddevice
python310Packages.soundfile
python310Packages.torchaudio-bin
python310Packages.transformers
portaudio
cudaPackages.cudatoolkit
stdenv.cc.cc.lib
binutils
gcc
] 
