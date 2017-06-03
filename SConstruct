vars = Variables()
vars.Add('DEBUG', '', 1)
vars.Add('FULL_DEBUG', '', 0)
vars.Add('NAME', '', 0)
vars.Add('ORG', '', 0)
vars.Add('NOISE', '', 0)
vars.Add('R0', '', 0)

env = Environment(variables = vars)
env.Append(CPPDEFINES={
	'DEBUG' : '${DEBUG}', 
	'FULL_DEBUG' : '${FULL_DEBUG}',
	'NAME' : '${NAME}',
	'ORG' : '${ORG}',
	'NOISE' : '${NOISE}',
	'R0' : '${R0}'})

env.Append(CPPPATH=['../pscrcl/src'])
env.Append(CPPPATH=['/usr/local/cuda/include'])

env.Append(CCFLAGS = ['-Wall', '-O2'])

env.Append(LIBPATH = ['../pscrcl/src/'])

env.Append(LIBS = ['OpenCL', 'pscrCL', 'png'])

common_sources = Split("""
V.cpp
Q.cpp
common.cpp
""")

env.Command("auglagL1gpu.cl.dat", "auglagL1gpu.cl", "xxd -i auglagL1gpu.cl > auglagL1gpu.cl.dat")

env.Command("../pscrcl/src/libpscrCL.a", "", "cd ../pscrcl/src/ && scons -Q DEBUG=${FULL_DEBUG} FULL_DEBUG=0")

env.Program(target='test', source=env.Object(source = [common_sources, "test.cpp", "io_png.c", "AuglagL1gpu.cpp"]))
