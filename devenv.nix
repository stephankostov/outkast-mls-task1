{ pkgs, lib, config, inputs, ... }:
let
  buildInputs = with pkgs; [
    cudaPackages_12_4.cuda_cudart
    cudaPackages_12_4.cudatoolkit
    cudaPackages_12_4.cudnn
    stdenv.cc.cc
    libuv
    zlib
  ];
in 
{
  packages = with pkgs; [
    cudaPackages_12_4.cuda_nvcc
  ];
  env = {
    LD_LIBRARY_PATH = "${with pkgs;lib.makeLibraryPath buildInputs}:/run/opengl-driver/lib:/run/opengl-driver-32/lib";
    CUDA_PATH = pkgs.cudaPackages_12_4.cudatoolkit;
  };

  languages.python = {
    version = "3.13";
    enable = true;
    uv = {
      enable = true;
      sync.enable = true;
    };
  };

  enterShell = ''
    echo "~ DEVENV ENVIRONMENT ACTIVATED ~"
    echo "Python path: $(which python)"
    echo "GLIBC version: $(ldd --version | head -n1)"
    echo "CUDA version: ${pkgs.cudaPackages_12_4.cudatoolkit.version}"
    . .devenv/state/venv/bin/activate
    echo "~ UV VENV ENVIRONMENT ACTIVATED ~"
    echo "Python path: $(which python)"
    echo "GLIBC version: $(ldd --version | head -n1)"
    echo "CUDA version: ${pkgs.cudaPackages_12_4.cudatoolkit.version}"
  '';
}