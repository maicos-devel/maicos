#/usr/bin/env bash
#
# Copyright (c) 2019 Authors and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
# SPDX-License-Identifier: GPL-2.0-or-later

_maicos_completion()
{
  local cur_word
  local prev_word
  local module
  local maicos_opts
  local maicos_default_args
  local topols
  local trajs

  cur_word="${COMP_WORDS[$COMP_CWORD]}"
  prev_word="${COMP_WORDS[$COMP_CWORD-1]}"
  module="${COMP_WORDS[1]}"

  #  The maicos options we will complete.
  maicos_opts=""
  maicos_opts+=" carbonstructure"
  maicos_opts+=" debye"
  maicos_opts+=" density_planar"
  maicos_opts+=" density_cylinder"
  maicos_opts+=" dielectric_spectrum"
  maicos_opts+=" dipole_angle"
  maicos_opts+=" diporder"
  maicos_opts+=" epsilon_bulk"
  maicos_opts+=" epsilon_planar"
  maicos_opts+=" epsilon_cylinder"
  maicos_opts+=" kinetic_energy"
  maicos_opts+=" saxs"
  maicos_opts+=" velocity"
  maicos_opts+=" --debug --help --bash_completion --version"

  maicos_default_args="-h -s -top -f -traj -atom_style -b -e -dt -box -nt -sel"

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
        -h|-top|-traj|-atomstyle|-b|-e|-dt|-box|-nt|-sel|-dout|-startq|-endq|-dq)
        COMPREPLY=( )
        return 0
        ;;
      esac
      COMPREPLY=( $( compgen -W "$maicos_default_args -dout -sq -startq \
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
        -h|-top|-traj|-atomstyle|-b|-e|-dt|-box|-nt|-sel|-dout|-dz|-temp|-zpos|-gr)
        COMPREPLY=( )
        return 0 ;;
      esac
      COMPREPLY=( $( compgen -W "$maicos_default_args -o -dout -d -dz -mu\
                                -muo -temp -zpos -dens" -- ${cur_word} ) )
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
          -h|-top|-traj|-atomstyle|-b|-e|-dt|-box|-nt|-sel|-dout|-dz|-temp|-zpos|-gr)
          COMPREPLY=( )
          return 0 ;;
        esac
        COMPREPLY=( $( compgen -W "$maicos_default_args -o -dout -d \
                                  -center -r -dr -l -dens" -- ${cur_word} ) )
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
      -h|-top|-traj|-atomstyle|-b|-e|-dt|-box|-nt|-sel|-temp|-o|-truncfac|-trunclen|-Nsegments|-noplots|-nobin)
      COMPREPLY=( )
      return 0 ;;
    esac
    COMPREPLY=( $( compgen -W "$maicos_default_args -recalc -temp -o -u \
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
      COMPREPLY=( $( compgen -W "$maicos_default_args -o -d \
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
        -h|-top|-traj|-atomstyle|-b|-e|-dt|-box|-nt|-sel|-dout|-dz|-sel|-shift)
        COMPREPLY=( )
        return 0 ;;
      esac
      COMPREPLY=( $( compgen -W "$maicos_default_args -o -dz -d \
                                -dout -sym -shift -com -bin -nopbcrepair" -- ${cur_word} ) )
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
        -h|-top|-traj|-atomstyle|-b|-e|-dt|-box|-nt|-sel|-dout|-sel|-temp)
        COMPREPLY=( )
        return 0 ;;
      esac
      COMPREPLY=( $( compgen -W "$maicos_default_args -o -dout -temp \
                                -nopbcrepair" -- ${cur_word} ) )
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
        -h|-top|-traj|-atomstyle|-b|-e|-dt|-box|-nt|-sel|-dout|-dz|-zmin|-zmax|-temp|-groups)
        COMPREPLY=( )
        return 0 ;;
      esac
      COMPREPLY=( $( compgen -W "$maicos_default_args -o -dout -dz -d -zmin \
                                -zmax -temp -groups -2d -vac -sym -com -nopbcrepair"\
                                                          -- ${cur_word} ) )
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
          COMPREPLY=( $( compgen -o plusdirs  -f -X "$topols" -- ${cur_word}) )
          return 0 ;;
          -o)
          COMPREPLY=( $( compgen -o plusdirs  -f -- ${cur_word}) )
          return 0 ;;
          -h|-top|-traj|-atomstyle|-b|-e|-dt|-box|-nt|-sel|-dout|-r|-dr-l|-dout|-temp)
          COMPREPLY=( )
          return 0 ;;
        esac
        COMPREPLY=( $( compgen -W "$maicos_default_args -o -dout -dr -vr -l \
                                  -temp -si -nopbcrepair" -- ${cur_word} ) )
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
        COMPREPLY=( $( compgen -W "$maicos_default_args -o " -- ${cur_word} ) )
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
          -h|-top|-traj|-atomstyle|-b|-e|-dt|-box|-nt|-sel|-startq|-endq|-dq|-mintheta|-maxtheta)
          COMPREPLY=( )
          return 0 ;;
        esac
        COMPREPLY=( $( compgen -W "$maicos_default_args -dout -sq\
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
            -dv)
            COMPREPLY=( $( compgen -W "0 1 2" -- ${cur_word}) )
            return 0 ;;
            -h|-top|-traj|-atomstyle|-b|-e|-dt|-box|-nt|-sel|-dout|-nbins|-nblock|-bpbc)
            COMPREPLY=( )
            return 0 ;;
          esac
          COMPREPLY=( $( compgen -W "$maicos_default_args -o -dout -d -dv -nbins\
                                    -nblock -nopbcrepair" -- ${cur_word} ) )
          return 0 ;;
  esac

  #  Complete the basic maicos commands.
  COMPREPLY=( $( compgen -W "${maicos_opts}" -- ${cur_word} ) )
  return 0
}

complete -o filenames -F _maicos_completion maicos
