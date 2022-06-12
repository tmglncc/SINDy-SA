PYTHON = python3
GNUPLOT = gnuplot

OUTPUT_FOLDER = output
OUTPUT_FILE = models.dat

.PHONY = run run_output clean
.DEFAULT_GOAL = run

run:
	mkdir -p ${OUTPUT_FOLDER}
	${PYTHON} Main.py
	${PYTHON} Pareto.py
	${GNUPLOT} sse.plt
	${GNUPLOT} weights.plt

run_output:
	mkdir -p ${OUTPUT_FOLDER}
	${PYTHON} Main.py > ${OUTPUT_FOLDER}/${OUTPUT_FILE}
	${PYTHON} Pareto.py
	${GNUPLOT} sse.plt
	${GNUPLOT} weights.plt

clean:
	rm -f ${OUTPUT_FOLDER}/*.png
	rm -f ${OUTPUT_FOLDER}/*.dat
	rm -f ${OUTPUT_FOLDER}/*.eps
	rm -f ${OUTPUT_FOLDER}/*.csv
