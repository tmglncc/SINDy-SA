set term postscript eps enhanced color font "Helvetica, 24" size 8,4
set colorsequence classic
set encoding utf8

set key vertical right top # spacing 1.2
set format y "%.3f"

set title "STLSQ - SSE"
set xlabel "Modelo"
set ylabel "SSE"

filename = 'output/weights.dat'

set auto x
set auto y

set style data histogram
set style histogram cluster gap 2
set style fill solid border -1
set boxwidth 0.7
set xtics scale 0

set output 'output/sse.eps'
plot filename using 2:xtic(1) notitle