#!/bin/bash
set -euo pipefail

release=v0.1.0
repo=https://github.com/christovnino-a11y/stt-lab
uv_version=0.13.7
uv_platform=uv-aarch64-apple-darwin
# https://github.com/astral-sh/uv/releases/expanded_assets/0.13.7
uv_sha256=81b811b57e4624df7ed3aac148e0359100d9962f3d9dce00aec533dc2e086731
python_version=3.12
project_files=(app.py pyproject.toml uv.lock)

usage() {
    cat <<'EOF'
Usage: install.sh [--install-only] [--source PATH]

Install STT Lab for the current user on Apple Silicon macOS and open it.
  --install-only  Install without starting the app.
  --source PATH   Read the application files from a local checkout.
  --help          Show this help.
EOF
}

fail() {
    printf 'STT Lab: %s\n' "$*" >&2
    exit 1
}

download() {
    /usr/bin/curl --proto '=https' --proto-redir '=https' --tlsv1.2 \
        --fail --location --silent --show-error --retry 3 --connect-timeout 20 \
        --output "$2" "$1"
}

write_launcher() {
    printf '%s\n' '#!/bin/bash' 'set -euo pipefail'
    printf 'install_root=%q\nproject_dir=%q\nuv=%q\npython_version=%q\n' \
        "$install_root" "$project_dir" "$uv" "$python_version"
    cat <<'EOF'

export UV_PYTHON_INSTALL_DIR="$install_root/python"
export UV_CACHE_DIR="$install_root/cache"
export UV_MANAGED_PYTHON=1
export UV_PROJECT_ENVIRONMENT="$project_dir/.venv"
cd "$project_dir"
exec "$uv" run --locked --python "$python_version" --no-dev python app.py --open-browser "$@"
EOF
}

main() {
    local source_dir='' install_only=0 file
    while (( $# )); do
        case "$1" in
            --install-only) install_only=1; shift ;;
            --source)
                (( $# >= 2 )) && [[ -n "$2" ]] || fail '--source needs a directory path.'
                source_dir=$2
                shift 2
                ;;
            -h|--help) usage; return ;;
            *) fail "Unknown option $1. Use --help for usage." ;;
        esac
    done

    [[ "$(/usr/bin/uname -s)" == Darwin ]] || fail 'This installer requires macOS.'
    [[ "$(/usr/bin/uname -m)" == arm64 ]] ||
        fail 'This installer requires Apple Silicon and a native arm64 Terminal.'

    if [[ -n "$source_dir" ]]; then
        source_dir=$(CDPATH= cd -- "$source_dir" && pwd -P) || fail 'Cannot open the source directory.'
        for file in "${project_files[@]}"; do
            [[ -f "$source_dir/$file" ]] || fail "Missing source file $source_dir/$file."
        done
    fi

    install_root=${STT_LAB_INSTALL_DIR:-"$HOME/Library/Application Support/STT Lab"}
    [[ "$install_root" == /* ]] || fail 'STT_LAB_INSTALL_DIR must be an absolute path.'
    local uv_dir="$install_root/tools/uv/$uv_version"
    uv="$uv_dir/uv"
    local launcher="$install_root/stt-lab"
    mkdir -p "$uv_dir" || fail "Cannot create the installation directory $install_root."

    # Keep staging on the same filesystem as the installed files for atomic moves.
    staging=$(mktemp -d "$install_root/.install.XXXXXXXX") || fail 'Cannot stage the installation.'
    trap 'rm -rf -- "$staging"' EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    local staged_project="$staging/project"
    mkdir "$staged_project"

    if [[ -z "$source_dir" ]]; then
        printf 'Downloading STT Lab %s…\n' "$release"
        local source_archive="$staging/source.tar.gz"
        download "$repo/archive/refs/tags/$release.tar.gz" "$source_archive" ||
            fail "Could not download STT Lab $release. Check the release URL and your connection."
        local source_prefix="stt-lab-${release#v}"
        for file in "${project_files[@]}"; do
            /usr/bin/tar -xOf "$source_archive" "$source_prefix/$file" > "$staged_project/$file" ||
                fail "Could not read $file from the $release archive."
        done
    else
        printf 'Reading STT Lab source from %s…\n' "$source_dir"
        for file in "${project_files[@]}"; do
            cp "$source_dir/$file" "$staged_project/$file"
        done
    fi

    if [[ ! -x "$uv" ]]; then
        printf 'Downloading uv %s for Apple Silicon…\n' "$uv_version"
        local uv_archive="$staging/$uv_platform.tar.gz"
        local uv_url="https://releases.astral.sh/github/uv/releases/download/$uv_version/$uv_platform.tar.gz"
        download "$uv_url" "$uv_archive" || fail "Could not download uv $uv_version."
        local actual_sha256
        actual_sha256=$(/usr/bin/shasum -a 256 < "$uv_archive")
        [[ "${actual_sha256%% *}" == "$uv_sha256" ]] ||
            fail "The checksum for uv $uv_version did not match."
        /usr/bin/tar -xOf "$uv_archive" "$uv_platform/uv" > "$staging/uv" ||
            fail 'Could not extract the uv executable.'
        chmod 755 "$staging/uv"
        mv -f "$staging/uv" "$uv"
    fi

    # Different source contents get separate environments so a failed install
    # cannot invalidate a working launcher. Identical installs reuse their env.
    local digest
    digest=$(cd "$staged_project" && /usr/bin/shasum -a 256 "${project_files[@]}" | /usr/bin/shasum -a 256)
    project_dir="$install_root/releases/$release/${digest%% *}"
    mkdir -p "$project_dir"
    for file in "${project_files[@]}"; do
        mv -f "$staged_project/$file" "$project_dir/$file"
    done

    export UV_PYTHON_INSTALL_DIR="$install_root/python"
    export UV_CACHE_DIR="$install_root/cache"
    export UV_MANAGED_PYTHON=1
    export UV_PROJECT_ENVIRONMENT="$project_dir/.venv"
    cd "$project_dir"

    printf 'Preparing Python %s…\n' "$python_version"
    "$uv" python install "$python_version" --no-bin || fail "Could not provision Python $python_version."
    printf 'Installing the locked Python dependencies…\n'
    "$uv" sync --locked --python "$python_version" --no-dev ||
        fail 'Could not install dependencies. You can run this installer again to retry.'

    write_launcher > "$staging/stt-lab"
    chmod 755 "$staging/stt-lab"
    mv -f "$staging/stt-lab" "$launcher"
    printf 'Installed STT Lab %s.\nTo start it again, run:\n  %q\n' "$release" "$launcher"

    if (( ! install_only )); then
        rm -rf -- "$staging"
        trap - EXIT INT TERM
        exec "$launcher"
    fi
}

main "$@"
