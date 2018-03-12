from numpy.distutils.core import setup, Extension
import os

d = {}
if 'SARA_C_ROOT' in os.environ:
    d['MKLROOT'] = os.path.join(os.environ['SARA_C_ROOT'], 'composer_xe_2015.2.164/mkl')
    extra_link_args = "-Wl,--start-group {MKLROOT}/lib/intel64/libmkl_intel_lp64.a {MKLROOT}/lib/intel64/libmkl_core.a {MKLROOT}/lib/intel64/libmkl_intel_thread.a -Wl,--end-group -lpthread -lm -ldl -liomp5".format(**d).split(' ')
else:
    d['MKLROOT'] = "/opt/intel/compilers_and_libraries_2018.0.128/linux/mkl"
    extra_link_args = "-Wl,--start-group {MKLROOT}/lib/intel64_lin/libmkl_intel_lp64.a {MKLROOT}/lib/intel64_lin/libmkl_core.a {MKLROOT}/lib/intel64_lin/libmkl_intel_thread.a -Wl,--end-group -lpthread -lm -ldl -liomp5".format(**d).split(' ')
extra_compile_args = "-mkl -D_Float128=__float128 -qopenmp -I{MKLROOT}/include".format(**d).split(' ')
ext_modules = [ Extension('mkl_helper', sources = ['mkl_helper.c'], extra_link_args=extra_link_args, extra_compile_args=extra_compile_args)]

module1 = Extension('qlknn_intel',
#                    include_dirs=['/opt/intel/compilers_and_libraries_2017.4.196/linux/mkl/include/'],
                    library_dirs = ['{MKLROOT}/lib/intel64', 'libmkl_rt'],

                    include_dirs=['{MKLROOT}/include'],


                    sources = ['qlknnmodule.c'],
                    extra_link_args=extra_link_args, extra_compile_args=extra_compile_args)

setup (name = 'PackageName',
       version = '1.0',
       description = 'This is a demo package',
       ext_modules = [module1])

# Compile with sudo python setup.py config --compiler=intelem build_clib --compiler=intelem build_ext --compiler=intelem install

