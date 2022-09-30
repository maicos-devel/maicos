set terminal dumb

set xlabel "z coordinate (Å)"
set ylabel "⍴_m"

plot "density.dat" u 1:2 w l
