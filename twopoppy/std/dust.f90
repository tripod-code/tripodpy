subroutine a(sizemin, sizemax, sizeint, expcalc, sizes, Nr, Nm)
  ! Subroutine calculates the particle sizes.
  !
  ! Parameters
  ! ----------
  ! sizemin(Nr) : Minimal particle size
  ! sizemax(Nr) : Maximum particle size
  ! sizeint(Nr) : Intermediate particle size
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
  double precision, intent(in)  :: sizeint(Nr)
  double precision, intent(in)  :: expcalc(Nr)
  double precision, intent(out) :: sizes(Nr, Nm)
  integer,          intent(in)  :: Nr
  integer,          intent(in)  :: Nm

  integer :: i

  do i=1, Nr
    if(expcalc(i) .eq. -5) then
      sizes(i, 1) = &
      sizeint(i) * sizemin(i) / ( sizeint(i) - sizemin(i) ) * log( sizeint(i) / sizemin(i) )
      sizes(i, 2) = &
      sizemax(i) * sizeint(i) / ( sizemax(i) - sizeint(i) ) * log( sizemax(i) / sizeint(i) )
    else if(expcalc(i) .eq. -4) then
      sizes(i, 1) = ( sizeint(i) - sizemin(i) ) / log( sizeint(i) / sizemin(i) )
      sizes(i, 2) = ( sizemax(i) - sizeint(i) ) / log( sizemax(i) / sizeint(i) )
    else
      sizes(i, 1) = &
      ( expcalc(i) + 4.d0 ) / ( expcalc(i) + 5.d0 ) * ( sizeint(i)**( expcalc(i) + 5.d0 ) -&
      sizemin(i)**( expcalc(i) + 5.d0) ) / ( sizeint(i)**( expcalc(i) + 4.d0 ) - sizemin(i)&
      **( expcalc(i) + 4.d0) )
      sizes(i, 2) = &
      ( expcalc(i) + 4.d0 ) / ( expcalc(i) + 5.d0 ) * ( sizemax(i)**( expcalc(i) + 5.d0 ) -&
      sizeint(i)**( expcalc(i) + 5.d0) ) / ( sizemax(i)**( expcalc(i) + 4.d0 ) - sizeint(i)&
      **( expcalc(i) + 4.d0) )
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

subroutine expcalc(Sigma, sizemax, sizeint, expo, Nr, Nm)
  ! Subroutine calculates the particle size distribution exponent.
  ! 
  ! Parameters
  ! ----------
  ! Sigma (Nr, Nm) : Dust surface density
  ! sizemax(Nr) : Maximum particle size
  ! sizeint(Nr) : Intermediate particle size
  ! Nr : Number or radial grid cells
  ! Nm : Number of mass bins
  !
  ! Returns
  ! -------
  ! expo(Nr) : Calculated distribution exponent

  implicit none

  double precision, intent(in)  :: Sigma(Nr, Nm)
  double precision, intent(in)  :: sizemax(Nr)
  double precision, intent(in)  :: sizeint(Nr)
  double precision, intent(out) :: expo(Nr)
  integer,          intent(in)  :: Nr
  integer,          intent(in)  :: Nm
  
  integer :: i

  do i=1, Nr
    expo(i) = log( Sigma(i, 1) / Sigma(i, 2) ) / log( sizemax(i) / sizeint(i) ) - 4.d0
  end do

end subroutine expcalc

subroutine sizeint(sizemin, sizemax, intsize, Nr)
  ! Subroutine calculates the intermediate particle size.
  ! 
  ! Parameters
  ! ----------
  ! sizemin(Nr) : Minimum particle size
  ! sizemax(Nr) : Maximum particle size
  ! Nr : Number or radial grid cells
  !
  ! Returns
  ! -------
  ! intsize(Nr) : Intermediate particle size

  implicit none

  double precision, intent(in)  :: sizemin(Nr)
  double precision, intent(in)  :: sizemax(Nr)
  double precision, intent(out) :: intsize(Nr)
  integer,          intent(in)  :: Nr
  
  double precision :: onehalf = 1.d0/2.d0
  
  intsize = ( sizemin * sizemax )**onehalf

end subroutine sizeint

subroutine sizemean(sizemin, sizemax, expcalc, meansize, Nr)
  ! Subroutine calculates the mass-averaged particle sizes.
  !
  ! Parameters
  ! ----------
  ! sizemin(Nr) : Minimal particle size
  ! sizemax(Nr) : Maximum particle size
  ! expcalc(Nr) : Calculated distribution exponent
  ! Nr : Number of radial grid cells
  !
  ! Returns
  ! -------
  ! meansize(Nr) : Mass-averaged particle size

  implicit None

  double precision, intent(in)  :: sizemin(Nr)
  double precision, intent(in)  :: sizemax(Nr)
  double precision, intent(in)  :: expcalc(Nr)
  double precision, intent(out) :: meansize(Nr)
  integer,          intent(in)  :: Nr

  integer :: i

  do i=1, Nr
    if(expcalc(i) .eq. -5) then
      meansize(i) = &
      sizemax(i) * sizemin(i) / ( sizemax(i) - sizemin(i) ) * log( sizemax(i) / sizemin(i) )
    else if(expcalc(i) .eq. -4) then
      meansize(i) = ( sizemax(i) - sizemin(i) ) / log( sizemax(i) / sizemin(i) )
    else
      meansize(i) = &
      ( expcalc(i) + 4.d0 ) / ( expcalc(i) + 5.d0 ) * ( sizemax(i)**( expcalc(i) + 5.d0 ) -&
      sizemin(i)**( expcalc(i) + 5.d0) ) / ( sizemax(i)**( expcalc(i) + 4.d0 ) - sizemin(i)&
      **( expcalc(i) + 4.d0) )
    end if
  end do

end subroutine sizemean
