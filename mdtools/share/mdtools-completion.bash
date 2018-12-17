#/usr/bin/env bash

_mdtools_completion()
{
  local cur_word
  local prev_word
  local mdtools_opts
  local module
  local topols
  local trajs

  cur_word="${COMP_WORDS[$COMP_CWORD]}"
  prev_word="${COMP_WORDS[$COMP_CWORD-1]}"
  module="${COMP_WORDS[1]}"

  #  The mdtools options we will complete.
  mdtools_opts=""
  mdtools_opts+=" carbonstructure"
  mdtools_opts+=" debye"
  mdtools_opts+=" density_planar"
  mdtools_opts+=" density_cylinder"
  mdtools_opts+=" dielectric_spectrum"
  mdtools_opts+=" dipole_angle"
  mdtools_opts+=" diporder"
  mdtools_opts+=" epsilon_bulk"
  mdtools_opts+=" epsilon_cylinder"
  mdtools_opts+=" epsilon_planar"
  mdtools_opts+=" kinetic_energy"
  mdtools_opts+=" saxs"
  mdtools_opts+=" velocity"
  mdtools_opts+=" --debug --help --version"

  #  Define knowing topology, trajectory, structure formats.
  topols='!*@(.txyz|.top|.dms|.gsd|.crd\
          .parm7|.data|.minimal|.xpdb|.xml|.prmtop|.ent|.tpr|.gms|.gro|.pdb|\
          .history|.mmtf|.mol2|.psf|.pdbqt|.pqr|.arc|.config|.xyz)'
  trajs='!*@(.chain|.crd|.dcd|.config|\
         .history|.dms|.gms|.gro|.inpcrd|.restrt|.lammps|.data|.mol2|.pdb|\
         .ent|.xpdb|.pdbqt|.pqr|.trj|.mdcrd|.crdbox|.ncdf|.nc|.trr|.trz|.xtc|\
         .xyz|.txyz|.arc|.memory|.mmtf|.gsd|.dummy|.lammpstrj)'
  structs='!*@(.gro|.g96|.pdb|.brk|.ent|.esp|.tpr)'

  #  Complete the arguments to the module commands.
  case "$module" in
    debye)
      case "${prev_word}" in
        -s)
        COMPREPLY=( $( compgen -o plusdirs  -f -X "$topols" -- ${cur_word}) )
        return 0 ;;
        -f)
        COMPREPLY=( $( compgen -o plusdirs  -f -X "$trajs" -- ${cur_word}) )
        return 0 ;;
        -sq)
        COMPREPLY=( $( compgen -o plusdirs  -f -- ${cur_word}) )
        return 0 ;;
        -d)
        COMPREPLY=( $( compgen -o plusdirs  -f -- ${cur_word}) )
        return 0 ;;
        -b|-e|-dt|-box|-sel|-dout|-startq|-endq|-dq)
        COMPREPLY=( )
        return 0
        ;;
      esac
      COMPREPLY=( $( compgen -W "-h -s -f -b -e -dt -box -sel -dout -sq -startq \
                    -endq -dq -sinc -d" -- ${cur_word} ) )
      return 0 ;;

    density_planar)
      case "${prev_word}" in
        -s)
        COMPREPLY=( $( compgen -o plusdirs  -f -X "$topols" -- ${cur_word}) )
        return 0 ;;
        -f)
        COMPREPLY=( $( compgen -o plusdirs  -f -X "$trajs" -- ${cur_word}) )
        return 0 ;;
        -o|-muo)
        COMPREPLY=( $( compgen -o plusdirs  -f -- ${cur_word}) )
        return 0 ;;
        -dens)
        COMPREPLY=( $( compgen -W "mass number charge temp" -- ${cur_word}) )
        return 0 ;;
        -d)
        COMPREPLY=( $( compgen -W "0 1 2" -- ${cur_word}) )
        return 0 ;;
        -b|-e|-dt|-box|-dout|-dz|-temp|-zpos|-gr)
        COMPREPLY=( )
        return 0 ;;
      esac
      COMPREPLY=( $( compgen -W "-h -s -f -b -e -dt -box -o -dout -d \
                                -dz -muo -temp -zpos -dens -gr -top -atom_style" -- ${cur_word} ) )
      return 0 ;;

    density_cylinder)
      case "${prev_word}" in
        -s)
        COMPREPLY=( $( compgen -o plusdirs  -f -X "$topols" -- ${cur_word}) )
        return 0 ;;
        -f)
        COMPREPLY=( $( compgen -o plusdirs  -f -X "$trajs" -- ${cur_word}) )
        return 0 ;;
        -o|-muo)
        COMPREPLY=( $( compgen -o plusdirs  -f -- ${cur_word}) )
        return 0 ;;
        -dens)
        COMPREPLY=( $( compgen -W "mass number charge temp" -- ${cur_word}) )
        return 0 ;;
        -d)
        COMPREPLY=( $( compgen -W "0 1 2" -- ${cur_word}) )
        return 0 ;;
        -b|-e|-dt|-box|-o|—dout|-center|-r|-dr|-l|-gr)
        COMPREPLY=( )
        return 0 ;;
      esac
      COMPREPLY=( $( compgen -W "-h -s -f -b -e -dt -box -o —dout \
                                -center -r -dr -l -dens -gr -top -atom_style" -- ${cur_word} ) )
      return 0 ;;
  dielectric_spectrum)
    case "${prev_word}" in
      -s)
      COMPREPLY=( $( compgen -o plusdirs  -f -X "$topols" -- ${cur_word}) )
      return 0 ;;
      -f)
      COMPREPLY=( $( compgen -o plusdirs  -f -X "$trajs" -- ${cur_word}) )
      return 0 ;;
      -method)
      COMPREPLY=( $( compgen -W "1 2" -- ${cur_word}) )
      return 0 ;;
      -o)
      COMPREPLY=( $( compgen -o plusdirs  -f -- ${cur_word}) )
      return 0 ;;
      -plotformat)
      COMPREPLY=( $( compgen -W "png pdf ps eps svg" -- ${cur_word}) )
      return 0 ;;
      -b|-e|-dt|-box|-temp|-o|-truncfac|-trunclen|-Nsegments|-noplots|-nobin)
      COMPREPLY=( )
      return 0 ;;
    esac
    COMPREPLY=( $( compgen -W "-h -s -f -b -e -dt -box -recalc -temp -o -u \
                               -truncfac -trunclen -segs -df -noplots ⁠\
                               -plotformat -ymin -nobin" -- ${cur_word} ) )
    return 0 ;;

    dipole_angle)
      case "${prev_word}" in
        -s)
        COMPREPLY=( $( compgen -o plusdirs  -f -X "$topols" -- ${cur_word}) )
        return 0 ;;
        -f)
        COMPREPLY=( $( compgen -o plusdirs  -f -X "$trajs" -- ${cur_word}) )
        return 0 ;;
        -o)
        COMPREPLY=( $( compgen -o plusdirs  -f -- ${cur_word}) )
        return 0 ;;
        -d)
        COMPREPLY=( $( compgen -W "0 1 2" -- ${cur_word}) )
        return 0 ;;
        -b|-e|-dt|-box|-dout|-d|-sel)
        COMPREPLY=( )
        return 0 ;;
      esac
      COMPREPLY=( $( compgen -W "-h -s -f -b -e -dt -box -o -d -sel \
                                -dout" -- ${cur_word} ) )
      return 0 ;;

    diporder)
      case "${prev_word}" in
        -s)
        COMPREPLY=( $( compgen -o plusdirs  -f -X "$topols" -- ${cur_word}) )
        return 0 ;;
        -f)
        COMPREPLY=( $( compgen -o plusdirs  -f -X "$trajs" -- ${cur_word}) )
        return 0 ;;
        -o)
        COMPREPLY=( $( compgen -o plusdirs  -f -- ${cur_word}) )
        return 0 ;;
        -bin)
        COMPREPLY=( $( compgen -W "COM COC OXY" -- ${cur_word}) )
        return 0 ;;
        -d)
        COMPREPLY=( $( compgen -W "0 1 2" -- ${cur_word}) )
        return 0 ;;
        -b|-e|-dt|-box|-dout|-dz|-sel|-shift)
        COMPREPLY=( )
        return 0 ;;
      esac
      COMPREPLY=( $( compgen -W "-h -s -f -b -e -dt -box -o -dz -d -sel \
                                -dout -sym -shift -com -bin" -- ${cur_word} ) )
      return 0 ;;

    epsilon_bulk)
      case "${prev_word}" in
        -s)
        COMPREPLY=( $( compgen -o plusdirs  -f -X "$topols" -- ${cur_word}) )
        return 0 ;;
        -f)
        COMPREPLY=( $( compgen -o plusdirs  -f -X "$trajs" -- ${cur_word}) )
        return 0 ;;
        -o)
        COMPREPLY=( $( compgen -o plusdirs  -f -- ${cur_word}) )
        return 0 ;;
        -b|-e|-dt|-box|-dout|-sel|-temp)
        COMPREPLY=( )
        return 0 ;;
      esac
      COMPREPLY=( $( compgen -W "-h -s -f -b -e -dt -box -o -dout -temp \
                                -nopbcrepair" -- ${cur_word} ) )
      return 0 ;;

    epsilon_cylinder)
      case "${prev_word}" in
        -s)
        COMPREPLY=( $( compgen -o plusdirs  -f -X "$topols" -- ${cur_word}) )
        return 0 ;;
        -f)
        COMPREPLY=( $( compgen -o plusdirs  -f -X "$trajs" -- ${cur_word}) )
        return 0 ;;
        -g)
        COMPREPLY=( $( compgen -o plusdirs  -f -X "$trajs" -- ${cur_word}) )
        return 0 ;;
        -o)
        COMPREPLY=( $( compgen -o plusdirs  -f -- ${cur_word}) )
        return 0 ;;
        -b|-e|-dt|-box|-dout|-r|-dr|-l)
        COMPREPLY=( )
        return 0 ;;
      esac
      COMPREPLY=( $( compgen -W "-h -s -f -b -e -dt -box -o -dout -g -r -dr -vr \
                                -l -si" -- ${cur_word} ) )
      return 0 ;;

    epsilon_planar)
      case "${prev_word}" in
        -s)
        COMPREPLY=( $( compgen -o plusdirs  -f -X "$topols" -- ${cur_word}) )
        return 0 ;;
        -f)
        COMPREPLY=( $( compgen -o plusdirs  -f -X "$trajs" -- ${cur_word}) )
        return 0 ;;
        -o)
        COMPREPLY=( $( compgen -o plusdirs  -f -- ${cur_word}) )
        return 0 ;;
        -d)
        COMPREPLY=( $( compgen -W "0 1 2" -- ${cur_word}) )
        return 0 ;;
        -b|-e|-dt|-box|-dout|-dz|-zmin|-zmax|-temp|-groups)
        COMPREPLY=( )
        return 0 ;;
      esac
      COMPREPLY=( $( compgen -W "-h -s -f -b -e -dt -box -o -dout -dz -d -zmin \
                                -zmax -temp -groups -2d -vac -sym -com -nopbcrepair"\
                                                          -- ${cur_word} ) )
      return 0 ;;

      kinetic_energy)
        case "${prev_word}" in
          -s)
          COMPREPLY=( $( compgen -o plusdirs  -f -X "$topols" -- ${cur_word}) )
          return 0 ;;
          -f)
          COMPREPLY=( $( compgen -o plusdirs  -f -X "$trajs" -- ${cur_word}) )
          return 0 ;;
          -o)
          COMPREPLY=( $( compgen -o plusdirs  -f -- ${cur_word}) )
          return 0 ;;
          -b|-e|-dt|-box)
          COMPREPLY=( )
          return 0 ;;
        esac
        COMPREPLY=( $( compgen -W "-h -s -f -b -e -dt -box -o " -- ${cur_word} ) )
        return 0 ;;

      saxs)
        case "${prev_word}" in
          -s)
          COMPREPLY=( $( compgen -o plusdirs  -f -X "$topols" -- ${cur_word}) )
          return 0 ;;
          -f)
          COMPREPLY=( $( compgen -o plusdirs  -f -X "$trajs" -- ${cur_word}) )
          return 0 ;;
          -sq)
          COMPREPLY=( $( compgen -o plusdirs  -f -- ${cur_word}) )
          return 0 ;;
          -b|-e|-dt|-box|-dout|-sel|-startq|-endq|-dq|-mintheta|-maxtheta)
          COMPREPLY=( )
          return 0 ;;
        esac
        COMPREPLY=( $( compgen -W "-h -s -f -b -e -dt -box -sel -dout -sq\
                                  -startq -endq -dq -mintheta -maxtheta" -- ${cur_word} ) )
        return 0 ;;

      velocity)
        case "${prev_word}" in
          -s)
          COMPREPLY=( $( compgen -o plusdirs  -f -X "$topols" -- ${cur_word}) )
          return 0 ;;
          -f)
          COMPREPLY=( $( compgen -o plusdirs  -f -X "$trajs" -- ${cur_word}) )
          return 0 ;;
          -o)
          COMPREPLY=( $( compgen -o plusdirs  -f -- ${cur_word}) )
          return 0 ;;
          -d)
          COMPREPLY=( $( compgen -W "0 1 2" -- ${cur_word}) )
          return 0 ;;
          -b|-e|-dt|-box|-dout|-dv|-nbins|-gr|-nblock)
          COMPREPLY=( )
          return 0 ;;
        esac
        COMPREPLY=( $( compgen -W "-h -s -f -b -e -dt -box -o -dout -d -dv -nbins \
                                  -gr -nblock" -- ${cur_word} ) )
        return 0 ;;
  esac

  #  Complete the basic mdtools commands.
  COMPREPLY=( $( compgen -W "${mdtools_opts}" -- ${cur_word} ) )
  return 0
}

complete -o filenames -F _mdtools_completion mdtools
