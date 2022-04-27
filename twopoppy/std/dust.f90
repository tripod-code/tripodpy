subroutine calculate_a(smin, smax, xi, a, Nr, Nm)
  ! Subroutine calculates the particle sizes.
  ! a = [a0, a1, 0.5*amean, amean]
  !
  ! Parameters
  ! ----------
  ! smin(Nr) : Minimal particle size
  ! smax(Nr) : Maximum particle size
  ! xi(Nr) : Calculated distribution exponent
  ! Nr : Number of radial grid cells
  ! Nm : Number of mass bins
  !
  ! Returns
  ! -------
  ! a(Nr, Nm) : Particle sizes

  implicit None

  double precision, intent(in)  :: smin(Nr)
  double precision, intent(in)  :: smax(Nr)
  double precision, intent(in)  :: xi(Nr)
  double precision, intent(out) :: a(Nr, Nm)
  integer,          intent(in)  :: Nr
  integer,          intent(in)  :: Nm

  integer :: i
  double precision :: sint(Nr)
  double precision :: xip4(Nr)
  double precision :: xip5(Nr)
  double precision :: R(Nr)
  double precision :: dum

  sint(:) = sqrt(smin(:) * smax(:))
  xip4(:) = xi(:) + 4.d0
  xip5(:) = xi(:) + 5.d0
  R(:) = xip4(:) / xip5(:)

  do i=1, Nr
    if(smax(i) .eq. smin(i)) then

      a(i, :) = smin(i)

    else if(xi(i) .eq. -5.d0) then

      a(i, 1) = sint(i) * smin(i) / (sint(i) - smin(i)) * log(sint(i) / smin(i))
      a(i, 2) = smax(i) * sint(i) / (smax(i) - sint(i)) * log(smax(i) / sint(i))
      a(i, 4) = smax(i) * smin(i) / (smax(i) - smin(i)) * log(smax(i) / smin(i))
      a(i, 3) = 0.5d0 * a(i, 4)

    else if(xi(i) .eq. -4.d0) then

      a(i, 1) = (sint(i) - smin(i)) / log(sint(i) / smin(i))
      a(i, 2) = (smax(i) - sint(i)) / log(smax(i) / sint(i))
      a(i, 4) = (smax(i) - smin(i)) / log(smax(i) / smin(i))
      a(i, 3) = 0.5d0 * a(i, 4)

    else

      dum = sqrt(smin(i) / smax(i))
      a(i, 1) = R(i) * sint(i) * (1.d0 - dum**xip5(i)) / (1.d0 - dum**xip4(i))
      a(i, 2) = R(i) * sint(i) * (dum**(-xip5(i)) - 1.d0) / (dum**(-xip4(i)) - 1.d0)
      ! a(i, 1) = R(i) * (sint(i)**xip5(i) - smin(i)**xip5(i)) / (sint(i)**xip4(i) - smin(i)**xip4(i))
      ! a(i, 2) = R(i) * (smax(i)**xip5(i) - sint(i)**xip5(i)) / (smax(i)**xip4(i) - sint(i)**xip4(i))
      a(i, 4) = R(i) * (smax(i)**xip5(i) - smin(i)**xip5(i)) / (smax(i)**xip4(i) - smin(i)**xip4(i))
      a(i, 3) = 0.5d0 * a(i, 4)

    end if

  end do

end subroutine calculate_a


subroutine fi_adv(Sigma, v, r, ri, Fi, Nr, Nm_s, Nm_l)
  ! Function calculates the advective mass fluxes through the grid cell interfaces.
  ! Velocity v is linearly interpolated on grid cell interfaces with
  ! vi(1, :) = vi(2, :) and vi(Nr+1, :) = vi(Nr, :).
  !
  ! Parameters
  ! ----------
  ! Sigma(Nr, Nm) : Surface density
  ! v(Nr, Nm) : Radial velocity at grid cell centers
  ! r(Nr) : Radial grid cell centers
  ! ri(Nr+1) : Radial grid cell interfaces
  ! Nr : Number of radial grid cells
  ! Nm_s : Short number of mass bins
  ! Nm_l : Long number of mass bins
  !
  ! Returns
  ! -------
  ! Fi(Nr+1, Nm) : Flux through grid cell interfaces.

  implicit none

  double precision, intent(in)  :: Sigma(Nr, Nm_s)
  double precision, intent(in)  :: v(Nr, Nm_l)
  double precision, intent(in)  :: r(Nr)
  double precision, intent(in)  :: ri(Nr+1)
  double precision, intent(out) :: Fi(Nr+1, Nm_s)
  integer,          intent(in)  :: Nr
  integer,          intent(in)  :: Nm_s
  integer,          intent(in)  :: Nm_l

  double precision :: vi(Nr+1)
  integer :: i
  integer :: ir

  do i=1, Nm_s
    call interp1d(ri, r, v(:, i), vi, Nr)
    do ir=2, Nr
      Fi(ir, i) = Sigma(ir-1, i) * max(0.d0, vi(ir)) + Sigma(ir, i) * min(vi(ir), 0.d0)
    end do
    Fi(1, i) = Sigma(1, i) * min(vi(2), 0.d0)
    Fi(Nr+1, i) = Sigma(Nr, i) * max(0.d0, vi(Nr))
  end do

end subroutine fi_adv


subroutine fi_diff(D, SigmaD, SigmaG, St, u, r, ri, Fi, Nr, Nm_s, Nm_l)
  ! Subroutine calculates the diffusive dust fluxes at the grid cell interfaces.
  ! The flux at the boundaries is assumed to be constant.
  !
  ! Parameters
  ! ----------
  ! D(Nr, Nm) : Dust diffusivity
  ! SigmaD(Nr, Nm) : Dust surface densities
  ! SigmaG(Nr) : Gas surface density
  ! St(Nr, Nm) : Stokes number
  ! u(Nr) : Gas turbulent RMS velocity
  ! r(Nr) : Radial grid cell centers
  ! ri(Nr+1) : Radial grid cell interfaces
  ! Nr : Number of radial grid cells
  ! Nm_s : Short number of mass bins
  ! Nm_l : Long number of mass bins
  !
  ! Returns
  ! -------
  ! Fi(Nr+1, Nm) : Diffusive fluxes at grid cell interfaces

  implicit none

  double precision, intent(in)  :: D(Nr, Nm_l)
  double precision, intent(in)  :: SigmaD(Nr, Nm_s)
  double precision, intent(in)  :: SigmaG(Nr)
  double precision, intent(in)  :: St(Nr, Nm_l)
  double precision, intent(in)  :: u(Nr)
  double precision, intent(in)  :: r(Nr)
  double precision, intent(in)  :: ri(Nr+1)
  double precision, intent(out) :: Fi(Nr+1, Nm_s)
  integer,          intent(in)  :: Nr
  integer,          intent(in)  :: Nm_s
  integer,          intent(in)  :: Nm_l

  double precision :: Di(Nr+1, Nm_s)
  double precision :: eps(Nr, Nm_s)
  double precision :: gradepsi(Nr+1, Nm_s)
  double precision :: lambda
  double precision :: P
  double precision :: SigDi(Nr+1, Nm_s)
  double precision :: SigGi(Nr+1)
  double precision :: Sti(Nr+1, Nm_s)
  double precision :: ui(Nr+1)
  double precision :: w
  integer :: ir
  integer :: i

  Fi(:, :) = 0.d0

  call interp1d(ri, r, SigmaG, SigGi, Nr)
  call interp1d(ri, r, u, ui, Nr)

  do i=1, Nm_s
    call interp1d(ri(:), r(:), D(:, i), Di(:, i), Nr)
    call interp1d(ri(:), r(:), SigmaD(:, i), SigDi(:, i), Nr)
    eps(:, i) = SigmaD(:, i) / SigmaG(:)
    call interp1d(ri(:), r(:), St(:, i), Sti(:, i), Nr)
  end do

  do ir=2, Nr
    gradepsi(ir, :) = (eps(ir, :) - eps(ir-1, :)) / (r(ir) - r(ir-1))
  end do

  do i=1, Nm_s
    do ir=2, Nr
      w = ui(ir) * SigDi(ir, i) / (1.d0 + Sti(ir, i)**2)
      Fi(ir, i) = -Di(ir, i) * SigGi(ir) * gradepsi(ir, i)
      P = abs( Fi(ir, i) / w )
      lambda = ( 1.d0 + P ) / ( 1.d0 + P + P**2 )
      if(lambda .GT. HUGE(lambda)) then
        Fi(ir, i) = w
      else
        Fi(ir, i) = lambda * Fi(ir, i)
      end if
    end do
  end do

  Fi(   1, :) = Fi( 2, :)
  Fi(Nr+1, :) = Fi(Nr, :)

end subroutine fi_diff


subroutine calculate_m(a, rhos, fill, masses, Nr, Nm)
  ! Subroutine calculates the particle masses.
  !
  ! Parameters
  ! ----------
  ! a(Nr, Nm) : Particle sizes
  ! rhos(Nr, Nm) : Solid state density
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
  double precision, intent(in)  :: rhos(Nr, Nm)
  double precision, intent(in)  :: fill(Nr, Nm)
  double precision, intent(out) :: masses(Nr, Nm)
  integer,          intent(in)  :: Nr
  integer,          intent(in)  :: Nm

  integer :: i, j

  do i=1, Nr
    do j=1, Nm
      masses(i, j) = 4.d0 / 3.d0 * pi * rhos(i, j) * fill(i, j) * a(i, j)**3.d0
    end do
  end do

end subroutine calculate_m


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
  ! pf(Nr, Nm, Nm) : Fragmentation probability in [0, 1]
  !
  ! Notes
  ! -----
  ! The sticking probability is ps = 1 - pf

  use constants, only: pi

  implicit none

  double precision, intent(in)  :: vrel(Nr, Nm, Nm)
  double precision, intent(in)  :: vfrag(Nr)
  double precision, intent(out) :: pf(Nr, Nm, Nm)
  integer,          intent(in)  :: Nr
  integer,          intent(in)  :: Nm

  double precision :: dum
  double precision :: fac
  integer :: ir
  integer :: i
  integer :: j

  fac = sqrt(108.d0 / 8.d0) * 2.d0 / (9.d0 * pi)
  do i=1, Nm
    do j=1, i
      do ir=2, Nr-1
        dum = (vfrag(ir)/vrel(ir, j, i))**2
        pf(ir, j, i) = fac * (1.5d0*dum + 1.d0) * exp(-1.5d0*dum)
        pf(ir, i, j) = pf(ir, j, i)
      end do
    end do
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
  double precision, intent(in)  :: m(Nr, Nm)
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
        dum = min(sqrt(fac2(ir) * (m(ir, j) + m(ir, i)) / (m(ir, j) * m(ir, i))), cs(ir))
        vrel(ir, j, i) = dum
        vrel(ir, i, j) = dum
      end do
    end do
  end do

end subroutine vrel_brownian_motion


subroutine calculate_xi(smin, smax, Sigma, xicalc, Nr, Nm)
! Subroutine calculates the exponent of the particle size distribution.
!
! Parameters
! ----------
! smin(Nr)      : Minimum particle size
! smax(Nr)      : Maximum particle size
! Sigma(Nr, Nm) : Dust surface density
! Nr            : Number of radial grid cells
! Nm            : Number of mass bins
!
! Returns
! -------
! xicalc(Nr)    : Calculated particle size distribution exponent

implicit none

double precision, intent(in)  :: smin(Nr)
double precision, intent(in)  :: smax(Nr)
double precision, intent(in)  :: Sigma(Nr, Nm)
double precision, intent(out) :: xicalc(Nr)
integer,          intent(in)  :: Nr
integer,          intent(in)  :: Nm

integer :: i
double precision :: sint(Nr)

sint(:) = sqrt(smin(:) * smax(:))

do i=1, Nr
  if(smax(i) .eq. sint(i)) then
    xicalc(i) = -2.5d0
  else
    xicalc(i) = max(-20.d0, min(15.d0, log(Sigma(i, 2) / Sigma(i, 1)) / log(smax(i) / sint(i)) - 4.d0))
  end if
end do

end subroutine calculate_xi


subroutine jacobian_coagulation_generator(M0, M1, dat, row, col, Nr, Nm)
  ! Subroutine calculates the coagulation Jacobian at every radial grid cell except for the boundaries.
  !
  ! Parameters
  ! ----------
  ! M0(Nr) : First diagonal element of grid cell Jacobain
  ! M1(Nr) : Second diagonal element of grid cell Jacobian
  ! Nr : Number of radial grid cells
  ! Nm : Number of mass bins
  !
  ! Returns
  ! -------
  ! dat((Nr-2)*Nm*Nm) : Non-zero elements of Jacobian
  ! row((Nr-2)*Nm*Nm) : row location of non-zero elements
  ! col((Nr-2)*Nm*Nm) : column location of non-zero elements
  
  implicit none

  double precision, intent(in)  :: M0(Nr)
  double precision, intent(in)  :: M1(Nr)
  double precision, intent(out) :: dat((Nr-2)*Nm*Nm)
  integer,          intent(out) :: row((Nr-2)*Nm*Nm)
  integer,          intent(out) :: col((Nr-2)*Nm*Nm)
  integer,          intent(in)  :: Nr
  integer,          intent(in)  :: Nm

  double precision :: jac(Nr, Nm, Nm)

  integer :: ir
  integer :: i
  integer :: j
  integer :: k
  integer :: start

  ! Initialization
  jac(:, :, :) = 0.d0
  dat(:) = 0.d0
  row(:) = 0
  col(:) = 0

  ! Filling the grid cell Jacobian
  do ir=2, Nr-1
    jac(ir, 1, 1) =  M0(ir)
    jac(ir, 1, 2) = -M1(ir)
    jac(ir, 2, 1) = -M0(ir)
    jac(ir, 2, 2) =  M1(ir)
  end do

  ! Filling the data array
  k = 1
  do ir=2, Nr-1
    start = (ir-1)*Nm - 1
    do i=1, Nm
      do j=1, Nm
        dat(k) = jac(ir, i, j)
        row(k) = start + i
        col(k) = start + j
        k = k + 1
      end do
    end do
  end do

end subroutine jacobian_coagulation_generator

subroutine s_coag(a, dv, H, m, pfrag, pstick, Sigma, smin, smax, xifrag, xistick, S, Nr, Nm)
  ! Subroutine calculates the coagulation source terms.
  !
  ! Parameters
  ! ----------
  ! a(Nr, Nm) : Particle sizes of a0 and a1
  ! dv(Nr, Nm, Nm) : Relative velocities of a0 and a1
  ! H(Nr, Nm) : Dust scale heights
  ! m(Nr, Nm) : Particle masses
  ! pfrag(Nr, Nm, Nm) : Fragmentation probability
  ! pstick(Nr, Nm, Nm) : Sticking probability
  ! Sigma(Nr, Nm) : Dust surface densities
  ! smin(Nr) : Minimum particle size
  ! smax(Nr) : Maximum particle size
  ! xifrag(Nr) : Fragmentation exponent
  ! xistick(Nr) : Sticking exponent
  ! Nr : Number of radial grid cells
  ! Nm : Number of mass bins (only a0 and a1)
  !
  ! Returns
  ! -------
  ! S(Nr, Nm) : Coagulation source terms

  use constants, only: pi

  implicit none

  double precision, intent(in)  :: a(Nr, Nm)
  double precision, intent(in)  :: dv(Nr, Nm, Nm)
  double precision, intent(in)  :: H(Nr, Nm)
  double precision, intent(in)  :: m(Nr, Nm)
  double precision, intent(in)  :: pfrag(Nr, Nm, Nm)
  double precision, intent(in)  :: pstick(Nr, Nm, Nm)
  double precision, intent(in)  :: Sigma(Nr, Nm)
  double precision, intent(in)  :: smin(Nr)
  double precision, intent(in)  :: smax(Nr)
  double precision, intent(in)  :: xifrag(Nr)
  double precision, intent(in)  :: xistick(Nr)
  double precision, intent(out) :: S(Nr, Nm)
  integer,          intent(in)  :: Nr
  integer,          intent(in)  :: Nm

  integer :: i
  integer :: j
  double precision :: sig(Nr, Nm, Nm)
  double precision :: dot01(Nr)
  double precision :: dot10(Nr)
  double precision :: F(Nr)
  double precision :: sint(Nr)
  double precision :: xiprime(Nr)

  ! Initialization
  sig(:, :, :) = 0.d0
  S(:, :) = 0.d0

  do i=1, Nm
    do j=1, Nm
      sig(:, j, i) = pi*(a(:, j)**2 + a(:, i)**2)
    end do
  end do

  sint(:) = sqrt(smin(:)*smax(:))

  xiprime(:) = pfrag(:, 2, 2)*xifrag(:) + pstick(:, 2, 2)*xistick(:)

  F(:) = H(:, 2) * sqrt(2.d0 / (H(:, 1)**2 + H(:, 2)**2)) &
    & * sig(:, 1, 2) / sig(:, 2, 2) * dv(:, 1, 2) / dv(:, 2, 2) &
    & * (smax(:)/sint(:))**(-xiprime(:) - 4.d0)

  dot01(:) = Sigma(:, 1) * Sigma(:, 2) * sig(:, 1, 2) * dv(:, 1, 2) / (m(:, 2) * sqrt(2.d0 * pi * (H(:, 1)**2 + H(:, 2)**2)))
  dot10(:) = Sigma(:, 2)**2 * sig(:, 2, 2) * dv(:, 2, 2) * F(:) / (2.d0 * sqrt(pi) * m(:, 2) * H(:, 2))

  S(:, 1) = dot10(:) - dot01(:)
  S(:, 2) = -S(:, 1)

end subroutine s_coag


subroutine smax_deriv(dv, rhod, rhos, smin, smax, vfrag, dsmax, Nr, Nm)
  ! Subroutine calculates the derivative of the maximum particle size
  !
  ! Parameters
  ! ----------
  ! dv(Nr) : Relative velocities of 1/2*<a> and <a>
  ! rhod(Nr, Nm) : Dust midplane volume densities
  ! rhos(Nr, Nm) : Particle bulk densities
  ! smin(Nr) : Minimum particle size
  ! smax(Nr) : Maximum particle size
  ! vfrag(Nr) : Fragmentation velocity
  ! Nr : Number of radial grid cells
  ! Nm : Number of mass bins (only a0 and a1)
  !
  ! Returns
  ! -------
  ! dsmax(Nr) : derivative of smax

  implicit none

  double precision, intent(in)  :: dv(Nr)
  double precision, intent(in)  :: rhod(Nr, Nm)
  double precision, intent(in)  :: rhos(Nr, Nm)
  double precision, intent(in)  :: smin(Nr)
  double precision, intent(in)  :: smax(Nr)
  double precision, intent(in)  :: vfrag(Nr)
  double precision, intent(out) :: dsmax(Nr)
  integer,          intent(in)  :: Nr
  integer,          intent(in)  :: Nm

  integer :: ir

  double precision :: A
  double precision :: B
  double precision :: f
  double precision :: rhod_sum
  double precision :: rhos_mean
  double precision :: thr

  ! Initialization
  dsmax(:) = 0.d0

  do ir=2, Nr-1

    A = (dv(ir) / vfrag(ir))**8
    B = (1.d0 - A) / (1.d0 + A)

    rhod_sum = SUM(rhod(ir, :))
    rhos_mean = SUM(rhod(ir, :)*rhos(ir, :)) / rhod_sum
    dsmax(ir) = rhod_sum / rhos_mean * dv(ir) * B

    if (dsmax(ir) .LT. 0.d0) then

      thr = 0.3d0 * smin(ir)
      f = 0.5d0 * (1.d0 + TANH(LOG10(smax(ir)/thr)/3.d-2))
      if( smax(ir) .LE. smin(ir) ) f = 0.d0
      dsmax(ir) = f * dsmax(ir)

    end if

  end do

end subroutine smax_deriv