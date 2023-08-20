{
  description = "Lore the Publicist Flake with a good example of python ml dependencies";
  inputs.nixpkgs.url = "github:nixos/nixpkgs/23.05";
 
  outputs = { self, nixpkgs }:
    let 
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system}.pkgs;
      lib = nixpkgs.lib;
    in {
      devShells.${system}.default = pkgs.mkShell {
        name = "Lore the Publicisit Environment";
        buildInputs = import ./nix-pkgs.nix pkgs;
        shellHook = "
          source ./nix-shell-entry.sh
        ";
      };
      dockerImage = pkgs.dockerTools.buildImage {
        name = "lore-the-publicist";
        tag = "latest";
        copyToRoot = [ "." ];
        config = {
          Cmd = [ "./lore" ];
        };
      };
  };
}
