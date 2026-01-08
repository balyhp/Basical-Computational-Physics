program twenty_four
    implicit none
    integer, parameter :: n = 4
    integer :: nums(n)
    character(len=1), dimension(4) :: ops = (/'+' , '-' , '*' , '/'/)
    logical :: found
    integer :: i

    print *, "Enter 4 numbers (1-13):"
    do i = 1, n
        read(*,*) nums(i)
        if (nums(i) < 1 .or. nums(i) > 13) then
            print *, "Invalid input."
            stop
        end if
    end do

    found = .false.
    call search(nums, ops, found)

    if (.not. found) print *, "No solution."
contains
    !===========================================
    subroutine search(nums, ops, found)
        implicit none
        integer, intent(in) :: nums(4)
        character(len=1), intent(in) :: ops(4)
        logical, intent(inout) :: found
        integer :: perm(4)
        logical :: used(4)

        perm = 0
        used = .false.
        call permute(nums, perm, used, 1, ops, found)
    end subroutine search

    recursive subroutine permute(nums, perm, used, depth, ops, found)
        implicit none
        integer, intent(in) :: nums(4)
        integer, intent(inout) :: perm(4)
        logical, intent(inout) :: used(4)
        integer, intent(in) :: depth
        character(len=1), intent(in) :: ops(4)
        logical, intent(inout) :: found
        integer :: i

        if (depth > 4) then
            call try_ops(perm, ops, found)
            return
        end if

        do i = 1, 4
            if (.not. used(i)) then
                used(i) = .true.
                perm(depth) = nums(i)
                call permute(nums, perm, used, depth+1, ops, found)
                used(i) = .false.
            end if
        end do
    end subroutine permute

    subroutine try_ops(nums, ops, found)
        implicit none
        integer, intent(in) :: nums(4)
        character(len=1), intent(in) :: ops(4)
        logical, intent(inout) :: found
        integer :: i, j, k

        do i = 1, 4
            do j = 1, 4
                do k = 1, 4
                    call check_forms(nums, (/ops(i), ops(j), ops(k)/), found)
                end do
            end do
        end do
    end subroutine try_ops

    subroutine check_forms(nums, o, found)
        implicit none
        integer, intent(in) :: nums(4)
        character(len=1), intent(in) :: o(3)
        logical, intent(inout) :: found
        real(8) :: a,b,c,d,r1
        real(8), parameter :: target = 24.0d0, tol = 1.0d-7

        a = real(nums(1), kind=8)
        b = real(nums(2), kind=8)
        c = real(nums(3), kind=8)
        d = real(nums(4), kind=8)

        ! ((a o1 b) o2 c) o3 d
        r1 = calc(calc(calc(a,o(1),b),o(2),c),o(3),d)
        if (abs(r1-target)<tol) then
            print *, "Solution: ((", nums(1), o(1), nums(2), ")", o(2), nums(3), ")", o(3), nums(4), "= 24"
            found=.true.
        end if

        ! (a o1 (b o2 c)) o3 d
        r1 = calc(calc(a,o(1),calc(b,o(2),c)),o(3),d)
        if (abs(r1-target)<tol) then
            print *, "Solution: (", nums(1), o(1), "(", nums(2), o(2), nums(3), "))", o(3), nums(4), "= 24"
            found=.true.
        end if

        ! (a o1 b) o2 (c o3 d)
        r1 = calc(calc(a,o(1),b),o(2),calc(c,o(3),d))
        if (abs(r1-target)<tol) then
            print *, "Solution: (", nums(1), o(1), nums(2), ")", o(2), "(", nums(3), o(3), nums(4), ") = 24"
            found=.true.
        end if

        ! a o1 ((b o2 c) o3 d)
        r1 = calc(a,o(1),calc(calc(b,o(2),c),o(3),d))
        if (abs(r1-target)<tol) then
            print *, "Solution: ", nums(1), o(1), "((", nums(2), o(2), nums(3), ")", o(3), nums(4), ") = 24"
            found=.true.
        end if

        ! a o1 (b o2 (c o3 d))
        r1 = calc(a,o(1),calc(b,o(2),calc(c,o(3),d)))
        if (abs(r1-target)<tol) then
            print *, "Solution: ", nums(1), o(1), "(", nums(2), o(2), "(", nums(3), o(3), nums(4), ")) = 24"
            found=.true.
        end if

        ! a o1 (b o2 (c o3 d))
        r1 = calc(a,o(1),calc(b,o(2),calc(c,o(3),d)))
        if (abs(r1-target)<tol) then
            print *, "Solution: ", nums(1), o(1), "(", nums(2), o(2), "(", nums(3), o(3), nums(4), ")) = 24"
            found=.true.
        end if
    end subroutine check_forms

    real(8) function calc(a,op,b)
        implicit none
        real(8), intent(in) :: a,b
        character(len=1), intent(in) :: op
        select case(op)
        case('+')
            calc=a+b
        case('-')
            calc=a-b
        case('*')
            calc=a*b
        case('/')
            if (abs(b)>1e-12) then
                calc=a/b
            else
                calc=1.0e30
            end if
        case default
            calc=0.0d0
        end select
    end function calc
    !===========================================
end program twenty_four