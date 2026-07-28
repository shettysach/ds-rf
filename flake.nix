{
  description = "Local CPU dev env";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = {
    self,
    nixpkgs,
    rust-overlay,
  }: let
    systems = [
      "x86_64-linux"
      "aarch64-linux"
      "x86_64-darwin"
      "aarch64-darwin"
    ];
    forAllSystems = nixpkgs.lib.genAttrs systems;
  in {
    devShells = forAllSystems (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [(import rust-overlay)];
        };

        # Dora pins 1.92 in CI; newer toolchains currently break a dependency.
        rustToolchain = pkgs.rust-bin.stable."1.92.0".minimal;
      in {
        default = pkgs.mkShell ({
            packages = with pkgs;
              [
                uv
                ty
                ruff
                rustToolchain
                pkg-config
                openssl
                #
                libglvnd
                mesa
                libx11
                libxcursor
                libxext
                libxi
                libxinerama
                libxrandr
                libxrender
              ]
              ++ pkgs.lib.optionals pkgs.stdenv.isLinux (with pkgs; [
                clang
                mold
                udev
              ]);

            # Use Nix's OpenSSL instead of compiling the vendored copy.
            OPENSSL_NO_VENDOR = "1";

            shellHook = ''
              export PATH=$PATH:/home/sword/.cargo/bin
                    export MUJOCO_GL=''${MUJOCO_GL:-egl}
                    export PYOPENGL_PLATFORM=''${PYOPENGL_PLATFORM:-egl}
                    export __EGL_VENDOR_LIBRARY_FILENAMES=''${__EGL_VENDOR_LIBRARY_FILENAMES:-${pkgs.mesa}/share/glvnd/egl_vendor.d/50_mesa.json}
                    export LIBGL_DRIVERS_PATH=''${LIBGL_DRIVERS_PATH:-${pkgs.mesa}/lib/dri}
                    export MESA_LOADER_DRIVER_OVERRIDE=''${MESA_LOADER_DRIVER_OVERRIDE:-llvmpipe}
                    export GALLIUM_DRIVER=''${GALLIUM_DRIVER:-llvmpipe}
                    export MPLCONFIGDIR=''${MPLCONFIGDIR:-/tmp/matplotlib-$USER}
                    export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [
                pkgs.libglvnd
                pkgs.mesa
                pkgs.libx11
                pkgs.libxcursor
                pkgs.libxext
                pkgs.libxi
                pkgs.libxinerama
                pkgs.libxrandr
                pkgs.libxrender
              ]}:''${LD_LIBRARY_PATH:-}
            '';
          }
          // pkgs.lib.optionalAttrs pkgs.stdenv.isLinux {
            CARGO_BUILD_RUSTFLAGS = "-C linker=${pkgs.clang}/bin/clang -C link-arg=-fuse-ld=${pkgs.mold}/bin/mold";
          });
      }
    );
  };
}
