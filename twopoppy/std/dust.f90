subroutine a(amin, amax, aint, xicalc, sizes, Nr, Nm)
  ! Subroutine calculates the particle sizes.
  ! a = [a0, a1, 0.5*amean, amean]
  !
  ! Parameters
  ! ----------
  ! amin(Nr) : Minimal particle size
  ! amax(Nr) : Maximum particle size
  ! aint(Nr) : Intermediate particle size
  ! xicalc(Nr) : Calculated distribution exponent
  ! Nr : Number of radial grid cells
  ! Nm : Number of mass bins
  !
  ! Returns
  ! -------
  ! sizes(Nr, Nm) : Particle sizes

  implicit None

  double precision, intent(in)  :: amin(Nr)
  double precision, intent(in)  :: amax(Nr)
  double precision, intent(in)  :: aint(Nr)
  double precision, intent(in)  :: xicalc(Nr)
  double precision, intent(out) :: sizes(Nr, Nm)
  integer,          intent(in)  :: Nr
  integer,          intent(in)  :: Nm

  integer :: i

  do i=1, Nr
    if(xicalc(i) .eq. -5) then
      sizes(i, 1) = &
      aint(i) * amin(i) / ( aint(i) - amin(i) ) * log( aint(i) / amin(i) )
      sizes(i, 2) = &
      amax(i) * aint(i) / ( amax(i) - aint(i) ) * log( amax(i) / aint(i) )
      sizes(i, 4) = &
      amax(i) * amin(i) / ( amax(i) - amin(i) ) * log( amax(i) / amin(i) )
      sizes(i, 3) = 0.5d0 * sizes(i, 4)
    else if(xicalc(i) .eq. -4) then
      sizes(i, 1) = ( aint(i) - amin(i) ) / log( aint(i) / amin(i) )
      sizes(i, 2) = ( amax(i) - aint(i) ) / log( amax(i) / aint(i) )
      sizes(i, 4) = ( amax(i) - amin(i) ) / log( amax(i) / amin(i) )
      sizes(i, 3) = 0.5d0 * sizes(i, 4)
    else
      sizes(i, 1) = &
      ( xicalc(i) + 4.d0 ) / ( xicalc(i) + 5.d0 ) * ( aint(i)**( xicalc(i) + 5.d0 ) -&
      amin(i)**( xicalc(i) + 5.d0) ) / ( aint(i)**( xicalc(i) + 4.d0 ) - amin(i)&
      **( xicalc(i) + 4.d0) )
      sizes(i, 2) = &
      ( xicalc(i) + 4.d0 ) / ( xicalc(i) + 5.d0 ) * ( amax(i)**( xicalc(i) + 5.d0 ) -&
      aint(i)**( xicalc(i) + 5.d0) ) / ( amax(i)**( xicalc(i) + 4.d0 ) - aint(i)&
      **( xicalc(i) + 4.d0) )
      sizes(i, 4) = &
      ( xicalc(i) + 4.d0 ) / ( xicalc(i) + 5.d0 ) * ( amax(i)**( xicalc(i) + 5.d0 ) -&
      amin(i)**( xicalc(i) + 5.d0) ) / ( amax(i)**( xicalc(i) + 4.d0 ) - amin(i)&
      **( xicalc(i) + 4.d0) )
      sizes(i, 3) = 0.5d0 * sizes(i, 4)
    end if
  end do

end subroutine a

subroutine m(a, rho, fill, masses, Nr, Nm)
  ! Subroutine calculates the particle masses.
  !
  ! Parameters
  ! ----------
  ! a(Nr, Nm) : Particle sizes
  ! rho(Nr, Nm) : Particle bulk densities
  ! fill(Nr, Nm) : Filling factor
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
  double precision, intent(in)  :: fill(Nr, Nm)
  double precision, intent(out) :: masses(Nr, Nm)
  integer,          intent(in)  :: Nr
  integer,          intent(in)  :: Nm

  integer :: i, j

  do i=1, Nr
    do j=1, Nm
      masses(i, j) = 4.d0 / 3.d0 * pi * rho(i, j) * fill(i, j) * a(i, j)**3.d0
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

subroutine xicalc(Sigma, amax, aint, xi, Nr, Nm)
  ! Subroutine calculates the particle size distribution exponent.
  !
  ! Parameters
  ! ----------
  ! Sigma (Nr, Nm) : Dust surface density
  ! amax(Nr) : Maximum particle size
  ! aint(Nr) : Intermediate particle size
  ! Nr : Number or radial grid cells
  ! Nm : Number of mass bins
  !
  ! Returns
  ! -------
  ! xi(Nr) : Calculated distribution exponent

  implicit none

  double precision, intent(in)  :: Sigma(Nr, Nm)
  double precision, intent(in)  :: amax(Nr)
  double precision, intent(in)  :: aint(Nr)
  double precision, intent(out) :: xi(Nr)
  integer,          intent(in)  :: Nr
  integer,          intent(in)  :: Nm

  integer :: i

  do i=1, Nr
    xi(i) = log( Sigma(i, 1) / Sigma(i, 2) ) / log( amax(i) / aint(i) ) - 4.d0
  end do

end subroutine xicalc

subroutine aint(amin, amax, intsize, Nr)
  ! Subroutine calculates the intermediate particle size.
  !
  ! Parameters
  ! ----------
  ! amin(Nr) : Minimum particle size
  ! amax(Nr) : Maximum particle size
  ! Nr : Number or radial grid cells
  !
  ! Returns
  ! -------
  ! intsize(Nr) : Intermediate particle size

  implicit none

  double precision, intent(in)  :: amin(Nr)
  double precision, intent(in)  :: amax(Nr)
  double precision, intent(out) :: intsize(Nr)
  integer,          intent(in)  :: Nr

  double precision :: onehalf = 1.d0/2.d0

  intsize = ( amin * amax )**onehalf

end subroutine aint
