program equation_opt
    implicit none
    integer, parameter :: n = 200
    integer :: a, b, i, j, e, npairs
    integer, allocatable :: pa(:), pb(:)          !  (a,b)
    integer(kind=8), allocatable :: sums(:)       !  a^5 + b^5
    integer(kind=8) :: pow5(0:n)
    integer(kind=8) :: e5, need, minSum, maxSum
    integer :: L, R, j0
    integer(kind=8), allocatable :: seen_keys(:)
    integer :: n_seen, cap_seen

    ! pre-compute x^5
    do i = 0, n
        pow5(i) = int(i, kind=8)**5
    end do

    ! generate (a,b) pairs and their sums
    npairs = (n+1)*(n+2)/2
    allocate(pa(npairs), pb(npairs), sums(npairs))

    j = 0
    do a = 0, n
        do b = a, n
            j = j + 1
            pa(j) = a
            pb(j) = b
            sums(j) = pow5(a) + pow5(b)
        end do
    end do

    ! sort according to sums and (a,b)
    call mergesort_pairs(sums, pa, pb, 1, npairs)
    
    ! initialize seen keys for uniqueness
    cap_seen = 1024
    allocate(seen_keys(cap_seen))
    n_seen = 0

    minSum = sums(1)
    maxSum = sums(npairs)

    print *, "Solutions to a^5 + b^5 + c^5 + d^5 = e^5 for a,b,c,d,e in [0,", n, "]"

    ! main loop O(n^3)
    do e = 0, n
        e5 = pow5(e)
        do i = 1, npairs
            need = e5 - sums(i)

            ! consider only  sums(i) <= need
            if (need < sums(i)) cycle

            if (need < minSum .or. need > maxSum) cycle

            call equal_range(sums, npairs, need, L, R)
            if (L <= R) then
                if (need == sums(i)) then
                    j0 = max(L, i)
                else
                    j0 = L
                end if
                do j = j0, R
                    ! output unique solution
                    call output_unique_simple(pa(i), pb(i), pa(j), pb(j), e, seen_keys, n_seen, cap_seen)
                end do
            end if
        end do
    end do

    deallocate(pa, pb, sums)

contains
    subroutine output_unique_simple(a, b, c, d, e, seen_keys, n_seen, cap_seen)
        implicit none
        integer, intent(in) :: a, b, c, d, e
        integer(kind=8), allocatable ,intent(inout) :: seen_keys(:)
        integer, intent(inout) :: n_seen, cap_seen
        integer :: x1, x2, x3, x4, idx
        integer(kind=8) :: key

        x1=a; x2=b; x3=c; x4=d
        call sort4ints(x1, x2, x3, x4)          ! listing
        key = pack_key4(x1, x2, x3, x4)         ! package as key

        do idx = 1, n_seen                      ! check duplication
            if (seen_keys(idx) == key) return
        end do

        if (n_seen == cap_seen) call grow_keys(seen_keys, cap_seen)
        n_seen = n_seen + 1
        seen_keys(n_seen) = key

        print *, " a=", x1, " b=", x2, " c=", x3, " d=", x4, " e=", e
    end subroutine output_unique_simple

   subroutine grow_keys(seen_keys, cap_seen)
        implicit none
        integer(kind=8), allocatable, intent(inout) :: seen_keys(:)
        integer, intent(inout) :: cap_seen
        integer(kind=8), allocatable :: tmp(:)
        allocate(tmp(cap_seen*2))
        tmp(1:cap_seen) = seen_keys(1:cap_seen)
        deallocate(seen_keys)
        call move_alloc(tmp, seen_keys)
        cap_seen = cap_seen * 2
    end subroutine grow_keys

    subroutine sort4ints(a, b, c, d)
        implicit none
        integer, intent(inout) :: a, b, c, d
        call iswap(a,b); call iswap(c,d)
        call iswap(a,c); call iswap(b,d)
        call iswap(b,c)
    end subroutine sort4ints

    subroutine iswap(x, y)
        implicit none
        integer, intent(inout) :: x, y
        integer :: t
        if (x > y) then
            t = x; x = y; y = t
        end if
    end subroutine iswap

   integer(kind=8) function pack_key4(a, b, c, d)
        implicit none
        integer, intent(in) :: a, b, c, d
        integer(kind=8), parameter :: base = 1000_8
        pack_key4 = (((int(a,8)*base + int(b,8))*base + int(c,8))*base + int(d,8))
    end function pack_key4


    recursive subroutine mergesort_pairs(s, a1, a2, l, r)
        integer(kind=8), intent(inout) :: s(:)
        integer, intent(inout)         :: a1(:), a2(:)
        integer, intent(in)            :: l, r
        integer :: m
        if (l >= r) return
        m = (l + r) / 2
        call mergesort_pairs(s, a1, a2, l, m)
        call mergesort_pairs(s, a1, a2, m+1, r)
        call merge_pairs(s, a1, a2, l, m, r)
    end subroutine mergesort_pairs

    subroutine merge_pairs(s, a1, a2, l, m, r)
        integer(kind=8), intent(inout) :: s(:)
        integer, intent(inout)         :: a1(:), a2(:)
        integer, intent(in)            :: l, m, r
        integer :: n1, n2, i, j, k
        integer(kind=8), allocatable :: ls(:), rs(:)
        integer, allocatable :: la1(:), la2(:), ra1(:), ra2(:)

        n1 = m - l + 1
        n2 = r - m
        allocate(ls(n1), la1(n1), la2(n1))
        allocate(rs(n2), ra1(n2), ra2(n2))

        ls = s(l:m);   la1 = a1(l:m);   la2 = a2(l:m)
        rs = s(m+1:r); ra1 = a1(m+1:r); ra2 = a2(m+1:r)

        i = 1; j = 1; k = l
        do while (i <= n1 .and. j <= n2)
            if (ls(i) <= rs(j)) then
                s(k) = ls(i); a1(k) = la1(i); a2(k) = la2(i)
                i = i + 1
            else
                s(k) = rs(j); a1(k) = ra1(j); a2(k) = ra2(j)
                j = j + 1
            end if
            k = k + 1
        end do
        do while (i <= n1)
            s(k) = ls(i); a1(k) = la1(i); a2(k) = la2(i)
            i = i + 1; k = k + 1
        end do
        do while (j <= n2)
            s(k) = rs(j); a1(k) = ra1(j); a2(k) = ra2(j)
            j = j + 1; k = k + 1
        end do

        deallocate(ls, la1, la2, rs, ra1, ra2)
    end subroutine merge_pairs

    subroutine equal_range(s, n, value, L, R)
        integer(kind=8), intent(in) :: s(:), value
        integer, intent(in) :: n
        integer, intent(out) :: L, R
        L = lower_bound(s, n, value)
        if (L == n+1 .or. s(L) /= value) then
            L = 1; R = 0
        else
            R = upper_bound(s, n, value)
        end if
    end subroutine equal_range

    integer function lower_bound(s, n, value)
        integer(kind=8), intent(in) :: s(:), value
        integer, intent(in) :: n
        integer :: lo, hi, mid
        lo = 1; hi = n + 1
        do while (lo < hi)
            mid = (lo + hi) / 2
            if (s(mid) < value) then
                lo = mid + 1
            else
                hi = mid
            end if
        end do
        lower_bound = lo
    end function lower_bound

    integer function upper_bound(s, n, value)
        integer(kind=8), intent(in) :: s(:), value
        integer, intent(in) :: n
        integer :: lo, hi, mid
        lo = 1; hi = n + 1
        do while (lo < hi)
            mid = (lo + hi) / 2
            if (s(mid) <= value) then
                lo = mid + 1
            else
                hi = mid
            end if
        end do
        upper_bound = lo - 1
    end function upper_bound
end program equation_opt
