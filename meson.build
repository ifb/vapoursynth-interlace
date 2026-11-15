project('interlace', 'c',
    default_options: ['buildtype=release', 'warning_level=2', 'b_lto=true', 'b_ndebug=if-release', 'c_std=c99'],
    license: 'LGPL-2.1-or-later',
    license_files: 'LICENSE',
    meson_version: '>=1.2.3',
    version: '2',
)

py = import('python').find_installation(pure: false)

incdir = include_directories(
    run_command(py, '-c', 'import vapoursynth as vs; print(vs.get_include())', check: true).stdout().strip(),
)

core_sources = files('src/cpu.c', 'src/vt_filter.c')

if host_machine.cpu_family() == 'x86_64'
    add_languages('nasm', native: false)
    nasm_args = ['-DARCH_X86_64=1', '-DPIC', '-Dprivate_prefix=p2i',
                 '-I' + meson.current_source_dir() / 'x86/']
    # mach-o mangles C symbols with a leading underscore; x86inc keys that on PREFIX
    if host_machine.system() == 'darwin'
        nasm_args += '-DPREFIX'
    endif
    add_project_arguments(nasm_args, language: 'nasm')
    # The asm picks its own tier at run time, so every build shares it.
    asm_libs = [static_library('interlace_asm',
        files('x86/cpu.asm', 'x86/vt_filter.asm'), pic: true)]
else
    asm_libs = []
endif

cc = meson.get_compiler('c')
core_c_args = cc.get_supported_arguments('-ffp-contract=off')

# The C is built once per x86 instruction-set level VapourSynth can
# choose from: with name.so listed in a manifest, it loads name.avx2.so,
# name.avx512.so or name.zn4.so when the CPU has the level and the file
# exists. The flags are the ones VapourSynth documents for each level.
cpu_levels = [['', []]]
if host_machine.cpu_family() == 'x86_64'
    cpu_levels += [
        ['.avx2', ['-march=x86-64-v3']],
        ['.avx512', ['-march=x86-64-v4']],
        ['.zn4', ['-march=znver4', '-mno-sse4a', '-mno-avx512bf16']],
    ]
endif

cores = []
foreach level : cpu_levels
    cores += static_library('interlace_core' + level[0].replace('.', '_'),
        core_sources,
        c_args: core_c_args + level[1],
        gnu_symbol_visibility: 'hidden',
        link_with: asm_libs,
        pic: true,
    )
endforeach
core = cores[0]

# One module serves both hosts, as ffms2 does: each looks up only its own
# entry point (VapourSynthPluginInit2 / avisynth_c_plugin_init), so the
# AviSynth frontend is simply another source file in the same target. It
# needs the vendored, pinned avisynth_c.h (interface v12).
plugin_sources = files('src/interlace.c')
plugin_incdirs = [incdir]

if get_option('avisynth')
    plugin_sources += files('src/interlace_avs.c')
    plugin_incdirs += include_directories('third_party/avisynth')
endif

# VapourSynth autoloads every module in the root of its plugins dir, but
# consults a manifest.vs only in a subdirectory, and only plugins listed
# in a manifest (or given to LoadPlugin) get the per-CPU builds picked
# for them.
plugin_install_dir = py.get_install_dir() / 'vapoursynth/plugins/interlace'

plugins = []
foreach i : range(cpu_levels.length())
    level = cpu_levels[i]
    plugins += shared_module('interlace' + level[0],
        plugin_sources,
        c_args: level[1],
        gnu_symbol_visibility: 'hidden',
        include_directories: plugin_incdirs,
        install: true,
        install_dir: plugin_install_dir,
        link_args: level[1],
        link_with: cores[i],
        name_prefix: '',
    )
endforeach
plugin = plugins[0]

install_data('manifest.vs', install_dir: plugin_install_dir)

# AviSynth+ end-to-end test. It links an external libavisynth, which the
# repo does not vendor and CI does not provide, so it registers only when
# -Davisynth_lib_dir points at one. Unlike the VapourSynth frontend the
# AviSynth side has no runtime dependency beyond the host (this filter does
# no resampling), so once the library is there the test runs unattended
# under `meson test`, cross-host comparison included.
avs_lib_dir = get_option('avisynth_lib_dir')
if avs_lib_dir != ''
    avisynth_dep = cc.find_library('avisynth', dirs: [avs_lib_dir])
    test_avs_exe = executable('test_interlace_avs',
        files('test/test_interlace_avs.c'),
        dependencies: avisynth_dep,
        include_directories: include_directories('third_party/avisynth'),
        build_rpath: avs_lib_dir,
    )
    # The cross-host comparison needs VapourSynth reference planes, which the
    # dumper writes by running the module through the VapourSynth frontend.
    vs_reference = custom_target('vs_reference',
        output: 'vsref',
        command: [py, files('test/dump_vs_reference.py'), plugin, '@OUTPUT@'],
        depends: plugin,
    )
    test('interlace_avs', test_avs_exe,
         args: [plugin.full_path(), vs_reference.full_path()],
         depends: [plugin, vs_reference])
endif

# Whole-frame benchmark: the per-row loop at real frame geometry, without a
# VapourSynth dependency, so it can be built static and run anywhere.
bench_frame = executable('bench_frame',
    files('test/bench_frame.c'),
    include_directories: include_directories('src'),
    link_with: core,
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
