subroutine calculate_a(smin, smax, q, fudge, a, Nr, Nm)
    ! Subroutine calculates the particle sizes.
    ! a = [a0, fudge * a1, a1, fudge * amax, amax]
    !
    ! This way this array can be passed to the dustpy function
    ! that computes relative velocities.
    !
    ! Parameters
    ! ----------
    ! smin(Nr) : Minimal particle size
    ! smax(Nr) : Maximum particle size
    ! q(Nr) : Calculated distribution exponent
    ! fudge : Fudge factor for vrel underlying smax evolution / fragmentation probability
    ! Nr : Number of radial grid cells
    ! Nm : Number of mass bins
    !
    ! Returns
    ! -------
    ! a(Nr, Nm) : Particle sizes

    implicit None

    double precision, intent(in) :: smin(Nr)
    double precision, intent(in) :: smax(Nr)
    double precision, intent(in) :: q(Nr)
    double precision, intent(in) :: fudge
    double precision, intent(out) :: a(Nr, Nm)
    integer, intent(in) :: Nr
    integer, intent(in) :: Nm

    integer :: i
    double precision :: sint(Nr)
    double precision :: qp4(Nr)
    double precision :: qp5(Nr)
    double precision :: qp6(Nr)
    double precision :: R1(Nr)
    double precision :: dum

    sint(:) = sqrt(smin(:) * smax(:))
    qp4(:) = q(:) + 4.d0
    qp5(:) = q(:) + 5.d0
    qp6(:) = q(:) + 6.d0
    R1(:) = qp4(:) / qp5(:)

    !#TODO: assumes flux-average = mass-average; might need update for Stokes-drag regime

    do i = 1, Nr
        if(q(i) == -5.d0) then
            a(i, 1) = sint(i) * smin(i) / (sint(i) - smin(i)) * log(sint(i) / smin(i))
            a(i, 3) = smax(i) * sint(i) / (smax(i) - sint(i)) * log(smax(i) / sint(i))
        else if(q(i) == -4.d0) then
            a(i, 1) = (sint(i) - smin(i)) / log(sint(i) / smin(i))
            a(i, 3) = (smax(i) - sint(i)) / log(smax(i) / sint(i))
        else
            dum = sqrt(smin(i) / smax(i))
            a(i, 1) = R1(i) * sint(i) * (1.d0 - dum**qp5(i)) / (1.d0 - dum**qp4(i))
            a(i, 3) = R1(i) * sint(i) * (dum**(-qp5(i)) - 1.d0) / (dum**(-qp4(i)) - 1.d0)
        end if
        a(i, 2) = fudge * a(i, 3)
        a(i, 4) = fudge * smax(i)
        a(i, 5) = smax(i)
    end do

end subroutine calculate_a


subroutine fi_diff(D, SigmaD, SigmaG, St, u, r, ri, Fi, Nr, Nm_s)
    ! Subroutine calculates the diffusive dust fluxes at the grid cell interfaces.
    ! The flux at the boundaries is assumed to be constant.
    !
    ! **NOTE**: here Stokes number and Diffusivity should be
    ! from a0 and a1 (mass average), so only a subset of the 5 size bins.
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
    !
    ! Returns
    ! -------
    ! Fi(Nr+1, Nm) : Diffusive fluxes at grid cell interfaces

    implicit none

    double precision, intent(in) :: D(Nr, Nm_s)
    double precision, intent(in) :: SigmaD(Nr, Nm_s)
    double precision, intent(in) :: SigmaG(Nr)
    double precision, intent(in) :: St(Nr, Nm_s)
    double precision, intent(in) :: u(Nr)
    double precision, intent(in) :: r(Nr)
    double precision, intent(in) :: ri(Nr + 1)
    double precision, intent(out) :: Fi(Nr + 1, Nm_s)
    integer, intent(in) :: Nr
    integer, intent(in) :: Nm_s

    double precision :: Di(Nr + 1, Nm_s)
    double precision :: eps(Nr, Nm_s)
    double precision :: gradepsi(Nr + 1, Nm_s)
    double precision :: lambda
    double precision :: P
    double precision :: SigDi(Nr + 1, Nm_s)
    double precision :: SigGi(Nr + 1)
    double precision :: Sti(Nr + 1, Nm_s)
    double precision :: ui(Nr + 1)
    double precision :: w
    integer :: ir
    integer :: i

    Fi(:, :) = 0.d0

    call interp1d(ri, r, SigmaG, SigGi, Nr)
    call interp1d(ri, r, u, ui, Nr)

    do i = 1, Nm_s
        call interp1d(ri(:), r(:), D(:, i), Di(:, i), Nr)
        call interp1d(ri(:), r(:), SigmaD(:, i), SigDi(:, i), Nr)
        eps(:, i) = SigmaD(:, i) / SigmaG(:)
        call interp1d(ri(:), r(:), St(:, i), Sti(:, i), Nr)
    end do

    do ir = 2, Nr
        gradepsi(ir, :) = (eps(ir, :) - eps(ir - 1, :)) / (r(ir) - r(ir - 1))
    end do

    do i = 1, Nm_s
        do ir = 2, Nr
            ! flux limiter:
            ! w is maximum diffusive flux
            ! Fi is the unlimted diffusive flux
            ! lambda limits Fi to w
            w = ui(ir) * SigDi(ir, i) / (1.d0 + Sti(ir, i)**2)
            Fi(ir, i) = -Di(ir, i) * SigGi(ir) * gradepsi(ir, i)
            P = abs(Fi(ir, i) / w)
            lambda = (1.d0 + P) / (1.d0 + P + P**2)
            if(lambda > HUGE(lambda)) then
                Fi(ir, i) = w
            else
                Fi(ir, i) = lambda * Fi(ir, i)
            end if
        end do
    end do

    Fi(1, :) = Fi(2, :)
    Fi(Nr + 1, :) = Fi(Nr, :)

end subroutine fi_diff


subroutine vrel_brownian_motion(cs, m, T,mu, vrel, Nr, Nm)
    ! Subroutine calculates the relative particle velocities due to Brownian motion.
    ! Its maximum value is the sound speed.
    !
    ! Parameters
    ! ----------
    ! cs(Nr) : Sound speed
    ! m(Nr, Nm) : Particle masses
    ! T(Nr) : Temperature
    ! mu(Nr) : mean molecular weight
    ! Nr : Number of radial grid cells
    ! Nm : Number of mass bins
    !
    ! Returns
    ! -------
    ! vrel(Nr, Nm, Nm) : Relative velocities

    use constants, only : k_B, pi

    implicit none

    double precision, intent(in) :: cs(Nr)
    double precision, intent(in) :: m(Nr, Nm)
    double precision, intent(in) :: T(Nr)
    double precision, intent(in) :: mu(Nr)
    double precision, intent(out) :: vrel(Nr, Nm, Nm)
    integer, intent(in) :: Nr
    integer, intent(in) :: Nm

    integer :: ir, i, j
    double precision :: fac1
    double precision :: fac2(Nr)
    double precision :: dum

    fac1 = 8.d0 * k_B / pi

    do ir = 1, Nr
        fac2(ir) = fac1 * T(ir)
    end do

    do i = 1, Nm
        do j = 1, i
            do ir = 1, Nr
                !dum = min(sqrt(fac2(ir) * (m(ir, j) + m(ir, i)) / (m(ir, j) * m(ir, i))), cs(ir))
                dum = (16. * cs(ir)**2 * mu(ir)/(pi*m(ir,i)))**0.5d0
                vrel(ir, j, i) = dum
                vrel(ir, i, j) = dum
            end do
        end do
    end do

end subroutine vrel_brownian_motion


subroutine vrel_radial_drift(vdriftmax, St,vrel, Nr, Nm)
  ! Subroutine calculates the relative particle velocities due to radial drift.
  !
  ! Parameters
  ! ----------
  ! vdriftmax(Nr) : Maximum drift velocity
  ! St(Nr, Nm) : Stokes number
  ! Nr : Number of radial grid cells
  ! Nm : Number of mass bins
  !
  ! Returns
  ! -------
  ! vrel(Nr, Nm) : Relative velocities

  implicit none

  double precision, intent(in)  :: vdriftmax(Nr)
  double precision, intent(in)  :: St(Nr,Nm)
  double precision, intent(out) :: vrel(Nr, Nm, Nm)
  integer,          intent(in)  :: Nr
  integer,          intent(in)  :: Nm
  
  integer :: ir, i, j
  double precision :: dum

  do i=1, Nm
    do j=1, i
      do ir=1, Nr
        dum = abs(2d0 * vdriftmax(ir) * (St(ir,i)/(St(ir,i)**2 +1d0) -  St(ir,j)/(St(ir,j)**2 +1d0)))
        vrel(ir, i, j) = dum
        vrel(ir, j, i) = dum
      end do
    end do
  end do

end subroutine vrel_radial_drift


subroutine vrel_vertical_settling(h, OmK, St, vrel, Nr, Nm)
  ! Subroutine calculates the relative particle velocities due to vertical settling.
  !
  ! Parameters
  ! ----------
  ! h(Nr, Nm) : Particle scale heights
  ! Omk(Nr) : Keplerian frequency
  ! St(Nr, Nm) : Stokes number
  ! Nr : Number of radial grid cells
  ! Nm : Number of mass bins
  !
  ! Returns
  ! -------
  ! vrel(Nr, Nm) : Relative velocities

  implicit none

  double precision, intent(in)  :: h(Nr, Nm)
  double precision, intent(in)  :: OmK(Nr)
  double precision, intent(in)  :: St(Nr, Nm)
  double precision, intent(out) :: vrel(Nr, Nm, Nm)
  integer,          intent(in)  :: Nr
  integer,          intent(in)  :: Nm

  integer :: ir, i, j

  double precision :: dum
  double precision :: OmMinSt(Nr, Nm)

  do i=1, Nm
    do ir=1, Nr
      OmMinSt(ir, i) = OmK(ir) * h(ir, i) * St(ir, i)/(St(ir, i) + 1d0)
    end do
  end do

  do i=1, Nm
    do j=1, i
      do ir=1, Nr
        dum = abs(OmMinSt(ir, j) - OmMinSt(ir, i))
        vrel(ir, j, i) = dum
        vrel(ir, i, j) = dum
      end do
    end do
  end do

end subroutine vrel_vertical_settling


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

    use constants, only : pi

    implicit None

    double precision, intent(in) :: a(Nr, Nm)
    double precision, intent(in) :: rhos(Nr, Nm)
    double precision, intent(in) :: fill(Nr, Nm)
    double precision, intent(out) :: masses(Nr, Nm)
    integer, intent(in) :: Nr
    integer, intent(in) :: Nm

    integer :: i, j

    do i = 1, Nr
        do j = 1, Nm
            masses(i, j) = 4.d0 / 3.d0 * pi * rhos(i, j) * fill(i, j) * a(i, j)**3.d0
        end do
    end do

end subroutine calculate_m


subroutine pfrag(vrel, vfrag, pf, Nr)
    ! Subroutine calculates the fragmentation probability.
    !
    ! Parameters
    ! ----------
    ! vrel(Nr) : total relative velocity between amax and 0.5 * amax
    ! vfrag(Nr) : Fragmentation velocity
    ! Nr : Number or radial grid cells
    !
    ! Returns
    ! -------
    ! pf(Nr) : Fragmentation probability in [0, 1]
    !
    ! Notes
    ! -----
    ! The sticking probability is ps = 1 - pf

    implicit none

    double precision, intent(in) :: vrel(Nr)
    double precision, intent(in) :: vfrag(Nr)
    double precision, intent(out) :: pf(Nr)
    integer, intent(in) :: Nr
    integer :: ir

    do ir = 1, Nr
        pf(ir) = exp(-(5.d0 * (min(vrel(ir) / vfrag(ir), 1d0) - 1d0))**2d0)
    end do

end subroutine pfrag

subroutine qfrag(p_dr, dv_tot, vfrag, St_max, q_turb1, q_turb2, &
    & q_drfr, alpha, SigmaGas, mump, q_frag, Nr)
    ! Subroutine calculates the power-law in the fragmentation
    ! regime, interpolating between different cases.
    !
    ! Note that the relative velocities passed to this subroutine
    ! should be the ...[:, -1, -2] entry, i.e. the relative ... velocities
    ! between a_max and half of a_max the drift component should also
    ! include the radial and azimuthal contributions.
    !
    ! Parameters
    ! ----------
    ! p_dr : the transition function between turbulence and drift
    ! dv_tot : the total relative velocities
    ! vfrag : the fragmentation velocity
    ! St_max : the maximum Stokes number
    ! q_turb1 : the power-law exponent for fragmentation in the first turbulence regime
    ! q_turb2 : same for the second turbulence regime
    ! alpha : the turbulence parameter
    ! SigmaGas : the gas surface density
    ! mump : array of mean molecular mass (\mu * m_p)
    ! q_drfr :  same if drift is causing fragmentation
    ! Nr : Number or radial grid cells
    !
    ! Returns
    ! -------
    ! q_frag(Nr) : Fragmentation power-law
    !
    use constants, only : sigma_H2,pi
    implicit none

    double precision, intent(in) :: p_dr(Nr)
    double precision, intent(in) :: dv_tot(Nr)
    double precision, intent(in) :: vfrag(Nr)
    double precision, intent(in) :: St_max(Nr)
    double precision, intent(in) :: q_turb1, q_turb2, q_drfr
    double precision, intent(in) :: alpha(Nr)
    double precision, intent(in) :: SigmaGas(Nr)
    double precision, intent(in) :: mump(Nr)
    double precision, intent(out) :: q_frag(Nr)
    integer, intent(in) :: Nr

    double precision :: Re, f_t1t2, p_t1, p_frag, q_turbfrag

    integer :: ir

    do ir = 1, Nr
        !This seems wrong with the paper
        Re =  0.5d0 * alpha(ir) * SigmaGas(ir) * sigma_H2 / mump(ir)

        ! Eq. A.1 of Pfeil+2024
        f_t1t2 = 5d0 * sqrt(1.d0 / Re) / St_max(ir)
        
        ! Eq. A.2
        p_t1 = 0.5d0 * (1d0 - (f_t1t2**4 - 1d0) / (f_t1t2**4 + 1d0))

        ! Eq. A.5
        p_frag = exp(-(5d0 * (min(dv_tot(ir) / vfrag(ir), 1d0) - 1d0))**2)

        q_turbfrag = p_t1 * q_turb1 + (1d0 - p_t1) * q_turb2

        q_frag(ir) = p_dr(ir) * q_drfr + (1d0 - p_dr(ir)) * q_turbfrag
    end do

end subroutine qfrag

subroutine pfrag_trans( St_max, alpha, SigmaGas, mump, p_frag_trans, Nr)
    ! Subroutine calculates the power-law in the fragmentation
    ! regime, interpolating between different cases.
    !
    ! Note that the relative velocities passed to this subroutine
    ! should be the ...[:, -1, -2] entry, i.e. the relative ... velocities
    ! between a_max and half of a_max the drift component should also
    ! include the radial and azimuthal contributions.
    !
    ! Parameters
    ! ----------
    ! St_max : the maximum Stokes number
    ! alpha : the turbulence parameter
    ! SigmaGas : the gas surface density
    ! mump : array of mean molecular mass (\mu * m_p)
    ! Nr : Number or radial grid cells
    !
    ! Returns
    ! -------
    ! p_frag_trans(Nr) : Fragmentation power-law
    !
    use constants, only : sigma_H2,pi
    implicit none

    double precision, intent(in) :: St_max(Nr)
    double precision, intent(in) :: alpha(Nr)
    double precision, intent(in) :: SigmaGas(Nr)
    double precision, intent(in) :: mump(Nr)
    double precision, intent(out) :: p_frag_trans(Nr)
    integer, intent(in) :: Nr

    double precision :: Re, f_t1t2

    integer :: ir

    do ir = 1, Nr
        !This seems wrong with the paper
        Re = 0.5d0 * alpha(ir) * SigmaGas(ir) * sigma_H2 / mump(ir)

        ! Eq. A.1 of Pfeil+2024
        f_t1t2 = 5d0 * sqrt(1.d0 / Re) / St_max(ir)
        
        ! Eq. A.2
        p_frag_trans(ir) = 0.5d0 * (1d0 - (f_t1t2**4 - 1d0) / (f_t1t2**4 + 1d0))
    end do

end subroutine pfrag_trans

subroutine pdriftfrag(dv_rad,dv_az, St_max, alpha, SigmaGas, mump, cs,p_frag_trans,p_driftfrag, Nr)
    ! Subroutine calculates the tranistion form the fragmentation to the drif tfragmentation limit
    !
    ! Parameters
    ! ----------
    ! dv_rad(Nr) : Radial relative velocities
    ! dv_az(Nr) : Azimuthal relative velocities
    ! St_max(Nr) : the maximum Stokes number
    ! alpha(Nr) : the turbulence parameter
    ! SigmaGas(Nr) : the gas surface density
    ! mump(Nr) : array of mean molecular mass (\mu * m_p)
    ! cs(Nr) : sound speed
    ! Nr : Number or radial grid cells
    ! Returns
    ! -------
    ! p_frag_trans(Nr) : transition probability framgnetation to drift fragmentation

    use constants, only : Sigma_H2
    implicit none

    double precision, intent(in) :: dv_rad(Nr)
    double precision, intent(in) :: dv_az(Nr)
    double precision, intent(in) :: St_max(Nr)
    double precision, intent(in) :: alpha(Nr)
    double precision, intent(in) :: SigmaGas(Nr)
    double precision, intent(in) :: mump(Nr)
    double precision, intent(in) :: cs(Nr)
    double precision, intent(in) :: p_frag_trans(Nr)
    double precision, intent(out) :: p_driftfrag(Nr)
    integer, intent(in) :: Nr


    !local variables
    double precision :: dv_drmax
    double precision :: St_mx, St_mn
    double precision :: Re
    double precision :: vgas, vsmall, vinter, v_tr_s, f_dt
    integer :: ir 


    do ir = 1 , Nr
        dv_drmax = max((dv_rad(ir)**2 + dv_az(ir)**2)**0.5d0, 1d-10)
        st_mx = St_max(ir)
        St_mn = 0.5d0 * St_max(ir)
        Re = 0.5d0 * alpha(ir) * SigmaGas(ir) * sigma_H2 / mump(ir)
        vgas = sqrt(1.5d0*alpha(ir)) *cs(ir)
        vsmall   = vgas * ((st_mx-st_mn)/(st_mx+st_mn) * (st_mx**2./(st_mx+Re**(-0.5d0)) - st_mn**2./(st_mn+Re**(-0.5d0))))**0.5d0
        vinter   = vgas * (2.292d0*st_mx)**0.5d0
        v_tr_s = p_frag_trans(ir)*vinter + (1.-p_frag_trans(ir))*vsmall

        f_dt = 0.3d0*v_tr_s/dv_drmax
        p_driftfrag(ir) =  0.5d0 + 0.5d0 * ((1.0d0 - f_dt**6) / (f_dt**6 + 1.0d0))
        if(p_driftfrag(ir) .ne. p_driftfrag(ir)) p_driftfrag(ir) = 0.0d0
    enddo 

end subroutine pdriftfrag


subroutine jacobian_coagulation_generator(sig, dv, H, m, Sigma, smin, smax, qeff, dat, row, col, Nr, Nm)
    ! Subroutine calculates the coagulation Jacobian at every radial grid cell except for the boundaries.
    !
    ! Parameters
    ! ----------
    ! sig(Nr, Nm) : collisional cross sections of (a0 and a1) and (a1 and fudge * a1)
    ! dv(Nr, Nm) : Relative velocities of (a0 and a1) and (a1 and fudge * a1)
    ! H(Nr, Nm) : Dust scale heights, of a0 and a1
    ! m(Nr, Nm) : Particle masses of a0 and a1
    ! Sigma(Nr, Nm) : Dust surface densities
    ! smin(Nr) : Minimum particle size
    ! smax(Nr) : Maximum particle size
    ! qeff(Nr) : size distribution exponent (computed from qturb1, ...)
    ! Nr : Number of radial grid cells
    ! Nm : Number of sizes, should be 2
    !
    ! Returns
    ! -------
    ! dat((Nr-2)*Nm*Nm) : Non-zero elements of Jacobian
    ! row((Nr-2)*Nm*Nm) : row location of non-zero elements
    ! col((Nr-2)*Nm*Nm) : column location of non-zero elements

    use constants, only : pi

    implicit none

    double precision, intent(in) :: sig(Nr, Nm)
    double precision, intent(in) :: dv(Nr, Nm)
    double precision, intent(in) :: H(Nr, Nm)
    double precision, intent(in) :: m(Nr, Nm)
    double precision, intent(in) :: Sigma(Nr, Nm)
    double precision, intent(in) :: smin(Nr)
    double precision, intent(in) :: smax(Nr)
    double precision, intent(in) :: qeff(Nr)
    double precision, intent(out) :: dat((Nr - 2) * Nm * Nm)
    integer, intent(out) :: row((Nr - 2) * Nm * Nm)
    integer, intent(out) :: col((Nr - 2) * Nm * Nm)
    integer, intent(in) :: Nr
    integer, intent(in) :: Nm

    double precision :: C1(Nr)
    double precision :: C2(Nr)
    double precision :: F(Nr)
    double precision :: jac(Nr, Nm, Nm)
    double precision :: M1(Nr)
    double precision :: M2(Nr)
    double precision :: sint(Nr)

    integer :: ir
    integer :: i
    integer :: j
    integer :: k
    integer :: start

    ! #TODO: here we do not include the shinkage term yet

    ! Initialization
    jac(:, :, :) = 0.d0
    dat(:) = 0.d0
    row(:) = 0
    col(:) = 0

    sint(:) = SQRT(smin(:) * smax(:))

    ! here collisions between large and small dust use a0 and a1
    ! which corresponds to a(1, 3) in the full size array (with helper sizes).
    F(:) = H(:, 2) * SQRT(2.d0 / (H(:, 1)**2 + H(:, 2)**2)) &
            & * sig(:, 1) / sig(:, 2) * dv(:, 1) / dv(:, 2) &
            & * (smax(:) / sint(:))**(-qeff(:) - 4.d0)

    C1(:) = sig(:, 1) * dv(:, 1) / (m(:, 2) * SQRT(2.d0 * pi * (H(:, 1)**2 + H(:, 2)**2)))
    C2(:) = sig(:, 2) * dv(:, 2) * F(:) / (2.d0 * m(:, 2) * SQRT(pi) * H(:, 2))
    !Jacobian of source term
    M1(:) = -C1(:) * Sigma(:, 2)
    M2(:) = -C2(:) * Sigma(:, 2)

    ! Filling the grid cell Jacobian
    jac(:, 1, 1) = M1(:)
    jac(:, 1, 2) = -M2(:)
    jac(:, 2, 1) = -M1(:)
    jac(:, 2, 2) = M2(:)

    ! Filling the data array
    k = 1
    do ir = 2, Nr - 1
        start = (ir - 1) * Nm - 1
        do i = 1, Nm
            do j = 1, Nm
                dat(k) = jac(ir, i, j)
                row(k) = start + i
                col(k) = start + j
                k = k + 1
            end do
        end do
    end do

end subroutine jacobian_coagulation_generator

subroutine s_coag(sig, dv, H, m, Sigma, smin, smax, qeff,Sigmin, S, Nr, Nm)
    ! Subroutine calculates the coagulation source terms.
    !
    ! Parameters
    ! ----------
    ! sig(Nr, Nm) : collisional cross sections of (a0 and a1) and (a1 and fudge * a1)
    ! dv(Nr, Nm) : Relative velocities of (a0 and a1) and (a1 and fudge * a1)
    ! H(Nr, Nm) : Dust scale heights, of a0 and a1
    ! m(Nr, Nm) : Particle masses of a0 and a1
    ! Sigma(Nr, Nm) : Dust surface densities
    ! smin(Nr) : Minimum particle size
    ! smax(Nr) : Maximum particle size
    ! qeff(Nr) : size distribution exponent (computed from qturb1, ...)
    ! Sigmin(Nr, Nm) : Dust surface densities
    ! Nr : Number of radial grid cells
    ! Nm : Number of mass bins (only a0 and a1)
    !
    ! Returns
    ! -------
    ! S(Nr, Nm) : Coagulation source terms

    use constants, only : pi

    implicit none

    double precision, intent(in) :: sig(Nr, Nm)
    double precision, intent(in) :: dv(Nr, Nm)
    double precision, intent(in) :: H(Nr, Nm)
    double precision, intent(in) :: m(Nr, Nm)
    double precision, intent(in) :: Sigma(Nr, Nm)
    double precision, intent(in) :: smin(Nr)
    double precision, intent(in) :: smax(Nr)
    double precision, intent(in) :: qeff(Nr)
    double precision, intent(in) :: Sigmin(Nr, Nm)
    double precision, intent(out) :: S(Nr, Nm)
    integer, intent(in) :: Nr
    integer, intent(in) :: Nm

    double precision :: dot01(Nr)
    double precision :: dot10(Nr)
    double precision :: F(Nr)
    double precision :: sint(Nr)

    ! Initialization
    S(:, :) = 0.d0

    sint(:) = sqrt(smin(:) * smax(:))

    ! here collisions between large and small dust use a0 and a1
    ! which corresponds to a(1, 3) in the full size array (with helper sizes).
    F(:) = H(:, 2) * SQRT(2.d0 / (H(:, 1)**2 + H(:, 2)**2)) &
            & * sig(:, 1) / sig(:, 2) * dv(:, 1) / dv(:, 2) &
            & * (smax(:) / sint(:))**(-qeff(:) - 4.d0)


    dot01(:) = Sigma(:, 1) * Sigma(:, 2) * sig(:, 1) * dv(:, 1) / (m(:, 2) * SQRT(2.d0 * pi * (H(:, 1)**2 + H(:, 2)**2)))
    dot10(:) = Sigma(:, 2)**2 * sig(:, 2) * dv(:, 2) * F(:) / (2.d0 * m(:, 2) * SQRT(pi) * H(:, 2))

    !#TODO: here we do not include the shinkage term yet
    where(sum(Sigma, dim=2) .gt. sum(Sigmin, dim=2))
        S(:, 1) = dot10(:) - dot01(:)
        S(:, 2) = -S(:, 1)
    elsewhere
        S(:, 1) = 0.d0
        S(:, 2) = 0.d0
    end where

end subroutine s_coag


subroutine smax_deriv(dv, rhod, rhos, smin, smax, vfrag, Sigma, SigmaFloor, &
        & dsmax, Nr, Nm)
    ! Subroutine calculates the derivative of the maximum particle size
    !
    ! Parameters
    ! ----------
    ! dv(Nr) : Relative velocities of 0.5 amax and amax
    ! rhod(Nr) : Dust midplane volume densities
    ! rhos(Nr) : Particle bulk densities
    ! smin(Nr) : Minimum particle size
    ! smax(Nr) : Maximum particle size
    ! vfrag(Nr) : Fragmentation velocity
    ! Sigma(Nr, Nm) : Dust surface density
    ! SigmaFloor(Nr, Nm) : Floor value of dust surface density
    ! Nr : Number of radial grid cells
    ! Nm : Number of mass bins (only a0 and a1)
    !
    ! Returns
    ! -------
    ! dsmax(Nr) : derivative of smax

    implicit none

    double precision, intent(in) :: dv(Nr)
    double precision, intent(in) :: rhod(Nr)
    double precision, intent(in) :: rhos(Nr)
    double precision, intent(in) :: smin(Nr)
    double precision, intent(in) :: smax(Nr)
    double precision, intent(in) :: vfrag(Nr)
    double precision, intent(in) :: Sigma(Nr, Nm)
    double precision, intent(in) :: SigmaFloor(Nr, Nm)
    double precision, intent(out) :: dsmax(Nr)
    integer, intent(in) :: Nr
    integer, intent(in) :: Nm

    integer :: ir

    double precision :: A
    double precision :: B
    double precision :: f
    double precision :: thr

    ! #TODO: here we do not include the shinkage term yet

    ! Initialization
    dsmax(:) = 0.d0

    do ir = 2, Nr - 1

        ! Prevents unwanted growth of smax
        if ( (Sigma(ir, 1) .lt. SigmaFloor(ir, 1)) .or. (Sigma(ir, 2) .lt. SigmaFloor(ir, 2))) then
            dsmax(ir) = 0.d0
        else

            A = (dv(ir) / vfrag(ir)) ** 3
            B = (1.d0 - A) / (1.d0 + A)
            B = 1.0d0 - 2.0d0 / (1.d0 + (vfrag(ir)/dv(ir))**3.d0)
            dsmax(ir) = rhod(ir) / rhos(ir) * dv(ir) * B

            ! limiter to stall negative growth near the lower size limit
            if (dsmax(ir) < 0.d0) then
                ! Factor 1.5 to ensure minimal distribution width
                thr = 1.5d0 * smin(ir)
                f = 0.5d0 * (1.d0 + TANH(LOG10(smax(ir) / thr) / 3.d-2))
                if(smax(ir) <= smin(ir)) f = 0.d0
                dsmax(ir) = f * dsmax(ir)
            end if

        end if

    end do

end subroutine smax_deriv

subroutine smax_deriv_shrink(dt, slim, f_crit, smax, Sig, sdot, nr, nm)
    ! Subroutine calculates the shrinkage source term.
    !
    ! Parameters
    ! ----------
    ! dt : time scale of shinakge for each radius
    ! slim : limiting size for shrinkage
    ! f_crit : mass fraction below which Sig1 should not drop
    ! smax : Maximum particle size
    ! Sig : Dust surface densities
    ! Nr : Number of radial grid cells
    !
    ! Returns
    ! -------
    ! sdot(Nr) : Shrinkage source term

    implicit none

    double precision, intent(in) :: dt(Nr)
    double precision, intent(in) :: slim
    double precision, intent(in) :: f_crit
    double precision, intent(in) :: smax(Nr)
    double precision, intent(in) :: Sig(Nr, Nm)
    double precision, intent(out) :: sdot(Nr)

    double precision :: t_dep(Nr)
    double precision :: eps_crit(Nr)
    double precision :: sdot_max(Nr)

    integer, intent(in) :: Nr, Nm

    eps_crit = f_crit * (Sig(:, 1) + Sig(:, 2))
    where (Sig(:, 2) .gt. f_crit * (Sig(:, 1) + Sig(:, 2)) .or. smax .le. slim)
        sdot = 0d0
    elsewhere
        t_dep = abs(Sig(:, 2)/(-(eps_crit - Sig(:,2))/dt))
        
        !the factor of 1+t_dep is differnt that in the paper
        sdot = smax /t_dep * (1d0 - smax / slim)
        !limitng factor to be changed
        sdot_max = 0.4*smax/dt 
        sdot =  sdot *sdot_max/sqrt(sdot**2 + sdot_max**2)
    end where

end subroutine smax_deriv_shrink

subroutine smax_deriv_shrink_2(dt, alim, q,  f_crit, amax,amin, Sig, sdot, nr, nm)
    ! Subroutine calculates the shrinkage source term.
    !
    ! Parameters
    ! ----------
    ! dt : time scale of shinakge for each radius
    ! slim : limiting size for shrinkage
    ! f_crit : mass fraction below which Sig1 should not drop
    ! smax : Maximum particle size
    ! Sig : Dust surface densities
    ! Nr : Number of radial grid cells
    !
    ! Returns
    ! -------
    ! sdot(Nr) : Shrinkage source term

    implicit none

    double precision, intent(in) :: dt
    double precision, intent(in) :: alim
    double precision, intent(in) :: q(Nr)
    double precision, intent(in) :: f_crit
    double precision, intent(in) :: amax(Nr)
    double precision, intent(in) :: amin(Nr)
    double precision, intent(in) :: Sig(Nr, Nm)
    double precision, intent(out) :: sdot(Nr)

    double precision :: Sig_crit(Nr)
    double precision :: Sig_tot(Nr)
    double precision :: dsig1dt(Nr)
    double precision :: sdot_max(Nr)
    double precision :: xi(Nr)
    double precision :: dum1(Nr)
    double precision :: dsig1da(Nr)

    integer, intent(in) :: Nr, Nm

    xi = q + 4d0   
    dum1 = (amax * amin)**(0.5*xi)
    Sig_tot = Sig(:, 1) + Sig(:, 2)
    Sig_crit = f_crit * (Sig(:, 1) + Sig(:, 2))
    dsig1dt = 0d0
    dsig1da = 0d0


    where (Sig(:, 2) .ge. Sig_crit .or. amax .le. alim .or. xi .eq. 0d0)    
        dsig1dt = 0d0
    elsewhere
        dsig1dt = (Sig_crit - Sig(:, 2)) / dt
    end where

    where (xi .eq. 0d0 .or. Sig(:, 2) .ge. Sig_crit .or. amax .le. alim)
        dsig1da = 0.0d0
    elsewhere
        dsig1da = abs(sig_tot * xi * (0.5d0 * amin**xi * dum1 + amax**xi * (0.5d0 * (dum1 - amin**xi))) &
        / (amax * (amax**xi - amin**xi)**2))
    end where

    sdot_max = 0.1*amax/dt

    where (dsig1da .ne. dsig1da)
        dsig1da = 0d0
    end where

    where(dsig1da .ne. 0d0)
        sdot = -dsig1dt/dsig1da
    elsewhere
        sdot = 0d0
    end where

    sdot  = sdot * sdot_max/sqrt(sdot**2 + sdot_max**2)


end subroutine smax_deriv_shrink_2

subroutine dadsig(alim, q,  f_crit, amax,amin, Sig, dadsig1, nr, nm)
    ! Subroutine calculates the shrinkage source term.
    !
    ! Parameters
    ! ----------
    ! dt : time scale of shinakge for each radius
    ! slim : limiting size for shrinkage
    ! f_crit : mass fraction below which Sig1 should not drop
    ! smax : Maximum particle size
    ! Sig : Dust surface densities
    ! Nr : Number of radial grid cells
    !
    ! Returns
    ! -------
    ! sdot(Nr) : Shrinkage source term

    implicit none

    double precision, intent(in) :: alim
    double precision, intent(in) :: q(Nr)
    double precision, intent(in) :: f_crit
    double precision, intent(in) :: amax(Nr)
    double precision, intent(in) :: amin(Nr)
    double precision, intent(in) :: Sig(Nr, Nm)
    double precision, intent(out) :: dadsig1(Nr)

    double precision :: Sig_crit(Nr)
    double precision :: Sig_tot(Nr)
    double precision :: dsig1dt(Nr)
    double precision :: sdot_max(Nr)
    double precision :: xi(Nr)
    double precision :: dum1(Nr)
    double precision :: dsig1da(Nr)

    integer, intent(in) :: Nr, Nm
    xi = q + 4d0    
    dum1 = (amax * amin)**(0.5*xi)
    Sig_tot = Sig(:, 1) + Sig(:, 2)
    Sig_crit = f_crit * (Sig(:, 1) + Sig(:, 2))
 
    dsig1dt = 0d0
    dsig1da = 0d0


    where (xi .eq. 0d0 .or. Sig(:, 2) .ge. Sig_crit .or. amax .le. alim)
        dsig1da = 0.0d0
    elsewhere
        dsig1da = abs(sig_tot * xi * (0.5d0 * amin**xi * dum1 + amax**xi * (0.5d0 * (dum1 - amin**xi))) &
        / (amax * (amax**xi - amin**xi)**2))
    end where

    where (dsig1da .ne. dsig1da)
        dsig1da = 0d0
    end where

    where(dsig1da .ne. 0d0)
        dadsig1 = -1./dsig1da
    elsewhere
        dadsig1 = 0d0
    end where

end subroutine dadsig

subroutine dsigda(alim, q,  f_crit, amax,amin, Sig, dsig1da, nr, nm)
    ! Subroutine calculates the shrinkage source term.
    !
    ! Parameters
    ! ----------
    ! dt : time scale of shinakge for each radius
    ! slim : limiting size for shrinkage
    ! f_crit : mass fraction below which Sig1 should not drop
    ! smax : Maximum particle size
    ! Sig : Dust surface densities
    ! Nr : Number of radial grid cells
    !
    ! Returns
    ! -------
    ! sdot(Nr) : Shrinkage source term

    implicit none

    double precision, intent(in) :: alim
    double precision, intent(in) :: q(Nr)
    double precision, intent(in) :: f_crit
    double precision, intent(in) :: amax(Nr)
    double precision, intent(in) :: amin(Nr)
    double precision, intent(in) :: Sig(Nr, Nm)
    double precision, intent(out) :: dsig1da(Nr)
    integer, intent(in) :: Nr, Nm

    double precision :: Sig_crit(Nr)
    double precision :: Sig_tot(Nr)
    double precision :: dsig1dt(Nr)
    double precision :: sdot_max(Nr)
    double precision :: xi(Nr)
    double precision :: dum1(Nr)



    xi = q + 4d0    
    dum1 = (amax * amin)**(0.5*xi)
    Sig_tot = Sig(:, 1) + Sig(:, 2)
    Sig_crit = f_crit * (Sig(:, 1) + Sig(:, 2))
 
    dsig1da = 0d0


    where (xi .eq. 0d0 .or. amax .le. alim)
        dsig1da = 0.0d0
    elsewhere
        dsig1da = (sig_tot * xi * (0.5d0 * amin**xi * dum1 + amax**xi * (0.5d0 * (dum1 - amin**xi))) &
        / (amax * (amax**xi - amin**xi)**2))
    end where

    where (dsig1da .ne. dsig1da)
        dsig1da = 0d0
    end where


end subroutine dsigda





subroutine sig_deriv_shrink(Sig, amin, amax,alim,q,dt,f_crit, adot_shrink, Sigdot_shrink, Nr, Nm)
    ! Subroutine calculates the derivative of the dust surface density
    ! caused by the shrinkage of the maximum particle size.
    !
    ! Parameters
    ! ----------
    ! Sig(Nr, Nm) : Dust surface densities
    ! amin(Nr) : Minimal particle size
    ! amax(Nr) : Maximum particle size
    ! xi(Nr) : Size distribution exponent (q+4)
    ! adot_shrink(Nr) : Shrinkage source term for the particle size
    ! Nr : Number of radial grid cells
    ! Nm : Number of size bins
    !
    ! Returns
    ! -------
    ! Sigdot_shrink(Nr) : Shrinkage source term

    implicit none

    double precision, intent(in) :: Sig(Nr, Nm)
    double precision, intent(in) :: amin(Nr)
    double precision, intent(in) :: amax(Nr)
    double precision, intent(in) :: alim
    double precision, intent(in) :: q(Nr)
    double precision, intent(in) :: dt
    double precision, intent(in) :: f_crit
    double precision, intent(in) :: adot_shrink(Nr)
    double precision, intent(out) :: Sigdot_shrink(Nr,Nm)

    integer, intent(in) :: Nr, Nm
    double precision :: dum2(Nr)
    double precision :: Sig_crit(Nr)
    double precision :: xi(Nr)
    double precision :: sig_tot(Nr)

    xi = q + 4d0
    sig_tot = Sig(:, 1) + Sig(:, 2)
    Sig_crit = f_crit * (Sig(:, 1) + Sig(:, 2))
    
    dum2 = 0 
    where (Sig(:, 2) .ge. Sig_crit .or. amax .le. alim .or. xi .eq. 0d0)
        dum2 = 0d0
    elsewhere
        dum2 = (Sig_crit - Sig(:, 2)) / dt
    end where


    Sigdot_shrink(:, 1) = -dum2
    Sigdot_shrink(:, 2) = dum2

end subroutine sig_deriv_shrink



subroutine jacobian_shrink_generator(Sigma, smin, smax,slim,  q, dt,f_crit,amax_dot, dat, row, col, Nr, Nm)
    ! Subroutine calculates the coagulation Jacobian at every radial grid cell except for the boundaries.
    !
    ! Parameters
    ! ----------
    ! Sigma(Nr, Nm) : Dust surface densities
    ! smin(Nr) : Minimum particle size
    ! smax(Nr) : Maximum particle size
    ! amax_dot(Nr) : drivative of max particle size
    ! Nr : Number of radial grid cells
    ! Nm : Number of sizes, should be 2
    !
    ! Returns
    ! -------
    ! dat((Nr-2)*Nm*Nm) : Non-zero elements of Jacobian
    ! row((Nr-2)*Nm*Nm) : row location of non-zero elements
    ! col((Nr-2)*Nm*Nm) : column location of non-zero elements

    use constants, only : pi

    implicit none

    double precision, intent(in) :: Sigma(Nr, Nm)
    double precision, intent(in) :: smin(Nr)
    double precision, intent(in) :: smax(Nr)
    double precision, intent(in) :: slim
    double precision, intent(in) :: q(Nr)
    double precision, intent(in) :: dt
    double precision, intent(in) :: f_crit
    double precision, intent(in) :: amax_dot(Nr)
    double precision, intent(out) :: dat((Nr - 2) * Nm * Nm)
    integer, intent(out) :: row((Nr - 2) * Nm * Nm)
    integer, intent(out) :: col((Nr - 2) * Nm * Nm)
    integer, intent(in) :: Nr
    integer, intent(in) :: Nm

    double precision :: jac(Nr, Nm, Nm)
    double precision :: xi(Nr)
    double precision :: dum1(Nr)
    double precision :: dum2(Nr)
    double precision :: Sig_crit(Nr)    


    integer :: ir
    integer :: i
    integer :: j
    integer :: k
    integer :: start

    ! #TODO: here we do not include the shinkage term yet

    ! Initialization
    jac(:, :, :) = 0.d0
    dat(:) = 0.d0
    row(:) = 0
    col(:) = 0

 
    xi = q(:) + 4.d0
    dum1 = (smax * smin)**(0.5*xi)
    ! here collisions between large and small dust use a0 and a1
    ! which corresponds to a(1, 3) in the full size array (with helper sizes).
    where(xi .eq. 0d0)
        dum2 = 0.0d0
    elsewhere
        dum2 =  xi * (0.5d0 * smin**xi * dum1 + smax**xi * (0.5d0 * (dum1 - smin**xi))) / (smax * (smax**xi - smin**xi)**2)
    end where

    where (dum2 .ne. dum2)
        dum2 = 0.0d0
    end where

    dum2 = dum2 * amax_dot

    dum2 = max(dum2, 0d0)

    Sig_crit = f_crit * (Sigma(:, 1) + Sigma(:, 2))

    dum2 = 0d0 
    dum1 = 0d0
    where (Sigma(:, 2) .ge. Sig_crit .or. smax .le. slim .or. xi .eq. 0d0)
        dum2 = 0d0
        dum1 = 0d0
    elsewhere
        dum2 = (f_crit - 1.) / dt
        dum1 = f_crit/dt 

    end where

    !check if the signs are correct
    jac(:, 1, 1) = -dum1
    jac(:, 2, 1) = dum1

    jac(:, 1, 2) = -dum2
    jac(:, 2, 2) = dum2

    ! Filling the data array
    k = 1
    do ir = 2, Nr - 1
        start = (ir - 1) * Nm - 1
        do i = 1, Nm
            do j = 1, Nm
                dat(k) = jac(ir, i, j)
                row(k) = start + i
                col(k) = start + j
                k = k + 1
            end do
        end do
    end do

end subroutine jacobian_shrink_generator

subroutine fi_diff_no_limit(D, SigmaD, SigmaG, St, u, r, ri, Fi, Nr, Nm)
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
    ! Nm : Number of mass bins
    !
    ! Returns
    ! -------
    ! Fi(Nr+1, Nm) : Diffusive fluxes at grid cell interfaces
  
  
    implicit none
  
    double precision, intent(in)  :: D(Nr, Nm)
    double precision, intent(in)  :: SigmaD(Nr, Nm)
    double precision, intent(in)  :: SigmaG(Nr)
    double precision, intent(in)  :: St(Nr, Nm)
    double precision, intent(in)  :: u(Nr)
    double precision, intent(in)  :: r(Nr)
    double precision, intent(in)  :: ri(Nr+1)
    double precision, intent(out) :: Fi(Nr+1, Nm)
    integer,          intent(in)  :: Nr
    integer,          intent(in)  :: Nm
  
    double precision :: Di(Nr+1, Nm)
    double precision :: eps(Nr, Nm)
    double precision :: gradepsi(Nr+1, Nm)
    double precision :: SigDi(Nr+1, Nm)
    double precision :: SigGi(Nr+1)
    double precision :: Sti(Nr+1, Nm)
    integer :: ir
    integer :: i
  
    Fi(:, :) = 0.d0
  
    call interp1d(ri, r, SigmaG, SigGi, Nr)
  
    do i=1, Nm
      call interp1d(ri(:), r(:), D(:, i), Di(:, i), Nr)
      call interp1d(ri(:), r(:), SigmaD(:, i), SigDi(:, i), Nr)
      eps(:, i) = SigmaD(:, i) / SigmaG(:)
      call interp1d(ri(:), r(:), St(:, i), Sti(:, i), Nr)
    end do
  
    do ir=2, Nr
      gradepsi(ir, :) = ( eps(ir, :) - eps(ir-1, :) ) / ( r(ir) - r(ir-1) )
    end do
  
    do i=1, Nm
        do ir=2, Nr
            Fi(ir, i) = -Di(ir, i) * SigGi(ir) * gradepsi(ir, i)
        enddo
    end do
  
    Fi(   1, :) = Fi( 2, :)
    Fi(Nr+1, :) = Fi(Nr, :)
  
  end subroutine fi_diff_no_limit
