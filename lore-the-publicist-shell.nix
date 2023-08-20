{ pkgs ? import <nixpkgs> {} }:
(pkgs.buildFHSUserEnv {
  name = "pipzone";
  targetPkgs = pkgs: (with pkgs; [
    python310
    python310Packages.pip
    python310Packages.virtualenv
    # Sound device depends on ctypes.util which depends on ld (not in python310 pkg)
    python310Packages.sounddevice
    portaudio
    cudaPackages.cudatoolkit
    stdenv.cc.cc.lib
    binutils
  ]);
  profile = ''
    source ./nix-shell-entry.sh
 '';
  runScript = "bash";
}).env
