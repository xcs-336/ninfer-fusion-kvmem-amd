@echo off
setlocal
set "ROCM_ROOT=D:\driver\ROCm\10.0.0"
call "D:\programfile\work\VS\18\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.44 > "%~dp0msvc-init.log" 2>&1
if errorlevel 1 exit /b 1
set "PATH=%ROCM_ROOT%\bin;%ROCM_ROOT%\lib\llvm\bin;%PATH%"
cmake -S "%~dp0." -B "%~dp0build" -G Ninja -DCMAKE_BUILD_TYPE=Release "-DCMAKE_CXX_COMPILER=%ROCM_ROOT%/lib/llvm/bin/clang++.exe" "-DROCM_ROOT=%ROCM_ROOT%" %*
if errorlevel 1 exit /b 1
cmake --build "%~dp0build" -j 2
if errorlevel 1 exit /b 1
ctest --test-dir "%~dp0build" --output-on-failure
exit /b %ERRORLEVEL%
