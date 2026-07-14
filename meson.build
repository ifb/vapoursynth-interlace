project('interlace', 'c',
    default_options: ['buildtype=release', 'warning_level=2', 'b_lto=true', 'b_ndebug=if-release', 'c_std=c99'],
    license: 'GPL-2.0-or-later',
    license_files: 'LICENSE',
    meson_version: '>=1.2.3',
    version: '1',
)

py = import('python').find_installation(pure: false)

incdir = include_directories(
    run_command(py, '-c', 'import vapoursynth as vs; print(vs.get_include())', check: true).stdout().strip(),
)

core_sources = files('src/cpu.c', 'src/vt_filter.c')

if host_machine.cpu_family() == 'x86_64'
    add_languages('nasm', native: false)
    add_project_arguments(
        '-DARCH_X86_64=1', '-DPIC', '-Dprivate_prefix=p2i',
        '-I' + meson.current_source_dir() / 'x86/',
        language: 'nasm',
    )
    core_sources += files('x86/cpu.asm', 'x86/vt_filter.asm')
endif

cc = meson.get_compiler('c')
core_c_args = cc.get_supported_arguments('-ffp-contract=off')

core = static_library('interlace_core', core_sources,
    c_args: core_c_args,
    gnu_symbol_visibility: 'hidden',
    pic: true,
)

shared_module('interlace',
    files('src/interlace.c'),
    gnu_symbol_visibility: 'hidden',
    include_directories: incdir,
    install: true,
    install_dir: py.get_install_dir() / 'vapoursynth/plugins',
    link_with: core,
    name_prefix: '',
)

checkasm_dep = dependency('checkasm', required: get_option('checkasm'), allow_fallback: true)
if checkasm_dep.found()
    checkasm_exe = executable('checkasm',
        files('test/checkasm/main.c', 'test/checkasm/vt_filter.c'),
        dependencies: checkasm_dep,
        include_directories: include_directories('src'),
        link_with: core,
    )
    test('checkasm', checkasm_exe)
endif
