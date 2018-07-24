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
  mdtools_opts+=" insert"
  mdtools_opts+=" debyer"
  mdtools_opts+=" density"
  mdtools_opts+=" density_cylinder"
  mdtools_opts+=" dielectric_spectrum"
  mdtools_opts+=" diporder"
  mdtools_opts+=" epsilon_bulk"
  mdtools_opts+=" epsilon_cylinder"
  mdtools_opts+=" epsilon_planar"
  mdtools_opts+=" pertop"
  mdtools_opts+=" saxs"
  mdtools_opts+=" velocity"
  mdtools_opts+=" --debug --help --version"

  #  Define knowing topology and trajectory formats.
  topols='!*@(.txyz|.top|.dms|.gsd|.crd\
          .parm7|.data|.minimal|.xpdb|.xml|.prmtop|.ent|.tpr|.gms|.gro|.pdb|\
          .history|.mmtf|.mol2|.psf|.pdbqt|.pqr|.arc|.config|.xyz)'
  trajs='!*@(.chain|.crd|.dcd|.config|\
         .history|.dms|.gms|.gro|.inpcrd|.restrt|.lammps|.data|.mol2|.pdb|\
         .ent|.xpdb|.pdbqt|.pqr|.trj|.mdcrd|.crdbox|.ncdf|.nc|.trr|.trz|.xtc|\
         .xyz|.txyz|.arc|.memory|.mmtf|.gsd|.dummy)'

  #  Complete the arguments to the module commands.
  case "$module" in
    carbonstructure)
      case "${prev_word}" in
        -s)
        COMPREPLY=( $( compgen -W "armcnt hopg hopgcnt" -- ${cur_word}) )
        return 0 ;;
        -i|-l|-w|-d|-x)
        COMPREPLY=( )
        return 0 ;;
      esac
      COMPREPLY=( $( compgen -W "-h -s -i -l -w -d -x -o" -- ${cur_word} ) )
      return 0 ;;

    insert)
      case "${prev_word}" in
        -o)
        COMPREPLY=( $( compgen -o plusdirs  -f -- ${cur_word}) )
        return 0 ;;
        -cp|-cs)
        COMPREPLY=( $( compgen -o plusdirs  -f -X "$trajs" -- ${cur_word}) )
        return 0 ;;
        -zmin|-zmax|-Nw|-d)
        COMPREPLY=( )
        return 0 ;;
      esac
      COMPREPLY=( $( compgen -W "-h -cp -cs -zmin -zmax -Nw -d -o" -- ${cur_word} ) )
      return 0 ;;

    debyer)
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
        -b|-e|-dt|-box|-sel|-dout|-startq|-endq|-dq|-d)
        COMPREPLY=( )
        return 0
        ;;
      esac
      COMPREPLY=( $( compgen -W "-h -s -f -b -e -dt -box -sel -dout -sq -startq \
                    -endq -dq -sinc -d" -- ${cur_word} ) )
      return 0 ;;

    density)
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
        -b|-e|-dt|-box|-dout|-d|-dz|-temp|-zpos|-dens|-gr)
        COMPREPLY=( )
        return 0 ;;
      esac
      COMPREPLY=( $( compgen -W "-h -s -f -b -e -dt -box -o -dout -d \
                                -dz -muo -temp -zpos -dens -gr" -- ${cur_word} ) )
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
        -b|-e|-dt|-box|-o|—dout|-center|-r|-dr|-l|-dens|-gr)
        COMPREPLY=( )
        return 0 ;;
      esac
      COMPREPLY=( $( compgen -W "-h -s -f -b -e -dt -box -o —dout \
                                -center -r -dr -l -dens -gr" -- ${cur_word} ) )
      return 0 ;;
  dielectric_spectrum)
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
      -b|-e|-dt|-box|-temp|-o|-truncfac|-trunclen|-Nsegments|-np)
      COMPREPLY=( )
      return 0 ;;
    esac
    COMPREPLY=( $( compgen -W "-h -s -f -b -e -dt -box -o -temp -truncfac\
                              -trunclen -Nsegments -np" -- ${cur_word} ) )
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
        -b|-e|-dt|-box|-dout|-d|-dz|-sel|-shift)
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
        -b|-e|-dt|-box|-dout|-dz|-d|-zmin|-zmax|-temp|-groups)
        COMPREPLY=( )
        return 0 ;;
      esac
      COMPREPLY=( $( compgen -W "-h -s -f -b -e -dt -box -o -dout -dz -d -zmin \
                                -zmax -temp -groups -2d -vac -sym -com -nopbcrepair"\
                                                          -- ${cur_word} ) )
      return 0 ;;

    pertop)
      case "${prev_word}" in
        -p)
        COMPREPLY=( $( compgen -o plusdirs  -f -X "!*@(.top|.itp)" -- ${cur_word}) )
        return 0 ;;
        -o)
        COMPREPLY=( $( compgen -o plusdirs  -f -- ${cur_word}) )
        return 0 ;;
        -l)
        COMPREPLY=( )
        return 0 ;;
      esac
      COMPREPLY=( $( compgen -W "-h -p -l -v" -- ${cur_word} ) )
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
          -b|-e|-dt|-box|-dout|-d|-dv|-nbins|-gr|-nblock)
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
