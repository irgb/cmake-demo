mkdir build
cd build

cmake ..
make

ctest -N -V

cd ..

COVERAGE_FILE=coverage.info
REPORT_FOLDER=coverage_report
lcov --rc lcov_branch_coverage=1 -c -d build -o ${COVERAGE_FILE}_tmp
lcov --rc lcov_branch_coverage=1  -e ${COVERAGE_FILE}_tmp "*hello*" -o ${COVERAGE_FILE}
genhtml --rc genhtml_branch_coverage=1 ${COVERAGE_FILE} -o ${REPORT_FOLDER}
rm -rf ${COVERAGE_FILE}_tmp
rm -rf ${COVERAGE_FILE}
