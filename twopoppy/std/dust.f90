subroutine a(sizemin, sizemax, expcalc, sizes, Nr, Nm)
  ! Subroutine calculates the particle sizes.
  !
  ! Parameters
  ! ----------
  ! sizemin(Nr) : Minimal particle size
  ! sizemax(Nr) : Maximum particle size
  ! expcalc(Nr) : Calculated distribution exponent
  ! Nr : Number of radial grid cells
  ! Nm : Number of mass bins
  !
  ! Returns
  ! -------
  ! sizes(Nr, Nm) : Particle sizes

  implicit None

  double precision, intent(in)  :: sizemin(Nr)
  double precision, intent(in)  :: sizemax(Nr)
  double precision, intent(in)  :: expcalc(Nr)
  double precision, intent(out) :: sizes(Nr, Nm)
  integer,          intent(in)  :: Nr
  integer,          intent(in)  :: Nm

  integer :: i
  double precision :: onehalf = 1.d0/2.d0
  double precision :: aint = (sizemin * sizemax)**onehalf

  do i=1, Nr
    if(expcalc .eq. -5) then
      sizes(i, 1) = aint(i) * sizemin(i) / ( aint(i) - sizemin(i) ) * log( aint(i) / sizemin(i) )
      sizes(i, 2) = sizemax(i) * aint(i) / ( sizemax(i) - aint(i) ) * log( sizemax(i) / aint(i) )
    else if(expcalc .eq. -4) then
      sizes(i, 1) = ( aint(i) - sizemin(i) ) / log( aint(i) / sizemin(i) )
      sizes(i, 2) = ( sizemax(i) - aint(i) ) / log( sizemax(i) / aint(i) )
    else
      sizes(i, 1) = &
      ( expcalc(i) + 4.d0 ) / ( expcalc(i) + 5.d0 ) * ( aint(i)**( expcalc(i) + 5.d0 ) -&
      sizemin(i)**( expcalc(i) + 5.d0) ) / ( aint(i)**( expcalc(i) + 4.d0 ) - sizemin(i)&
      **( expcalc(i) + 4.d0) )
      sizes(i, 2) = &
      ( expcalc(i) + 4.d0 ) / ( expcalc(i) + 5.d0 ) * ( sizemax(i)**( expcalc(i) + 5.d0 )&
      - aint(i)**( expcalc(i) + 5.d0) ) / ( sizemax(i)**( expcalc(i) + 4.d0 ) - aint(i)**&
      ( expcalc(i) + 4.d0) )
    end if
  end do

end subroutine a

subroutine m(a, rho, masses, Nr, Nm)
  ! Subroutine calculates the particle masses.
  !
  ! Parameters
  ! ----------
  ! a(Nr, Nm) : Particle sizes
  ! rho(Nr, Nm) : Particle bulk densities
  ! Nr : Number of radial grid cells
  ! Nm : Number of mass bins
  !
  ! Returns
  ! -------
  ! masses(Nr, Nm) : Particle masses

  use constants, only: pi

  implicit None

  double precision, intent(in)  :: a(Nr, Nm)
  double precision, intent(in)  :: rho(Nr, Nm)
  double precision, intent(out) :: masses(Nr, Nm)
  integer,          intent(in)  :: Nr
  integer,          intent(in)  :: Nm

  integer :: i, j

  do i=1, Nr
    do j=1, Nm
      masses(i, j) = 4.d0 / 3.d0 * pi * rho(i, j) * a(i, j)**3.d0
    end do
  end do

end subroutine m

subroutine pfrag(vrel, vfrag, pf, Nr, Nm)
  ! Subroutine calculates the fragmentation probability.
  ! It is assuming a Maxwell-Boltzmann velocity distribution.
  ! 
  ! Parameters
  ! ----------
  ! vrel(Nr, Nm, Nm) : Relative velocity
  ! vfrag(Nr) : Fragmentation velocity
  ! Nr : Number or radial grid cells
  ! Nm : Number of mass bins
  !
  ! Returns
  ! -------
  ! pf(Nr) : Fragmentation probability in [0, 1]
  !
  ! Notes
  ! -----
  ! The sticking probability is ps = 1 - pf

  implicit none

  double precision, intent(in)  :: vrel(Nr, Nm, Nm)
  double precision, intent(in)  :: vfrag(Nr)
  double precision, intent(out) :: pf(Nr)
  integer,          intent(in)  :: Nr
  integer,          intent(in)  :: Nm
  
  double precision :: dum
  integer :: i

  do i=1, Nr
    dum = (vfrag(i)/vrel(i, 2, 2))**2
    pf(i) = (1.5d0*dum + 1.d0) * exp(-1.5d0*dum)
  end do

end subroutine pfrag

subroutine vrel_brownian_motion(cs, m, T, vrel, Nr, Nm)
  ! Subroutine calculates the relative particle velocities due to Brownian motion.
  ! Its maximum value is the sound speed.
  !
  ! Parameters
  ! ----------
  ! cs(Nr) : Sound speed
  ! m(Nr, Nm) : Particle masses
  ! T(Nr) : Temperature
  ! Nr : Number of radial grid cells
  ! Nm : Number of mass bins
  !
  ! Returns
  ! -------
  ! vrel(Nr, Nm, Nm) : Relative velocities

  use constants, only: k_B, pi

  implicit none

  double precision, intent(in)  :: cs(Nr)
  double precision, intent(in)  :: m(Nm)
  double precision, intent(in)  :: T(Nr)
  double precision, intent(out) :: vrel(Nr, Nm, Nm)
  integer,          intent(in)  :: Nr
  integer,          intent(in)  :: Nm

  integer :: ir, i, j
  double precision :: fac1
  double precision :: fac2(Nr)
  double precision :: dum

  fac1 = 8.d0 * k_B / pi

  do ir=1, Nr
    fac2(ir) = fac1 * T(ir)
  end do

  do i=1, Nm
    do j=1, i
      do ir=1, Nr
        dum = min(sqrt(fac2(ir)*(m(ir, j)+m(ir, i))/(m(ir, j)*m(ir, i))), cs(ir))
        vrel(ir, j, i) = dum
        vrel(ir, i, j) = dum
      end do
    end do
  end do

end subroutine vrel_brownian_motion
