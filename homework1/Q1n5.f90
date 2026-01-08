! O(n**5) method

program equation
    implicit none
    integer, parameter :: n = 200
    integer :: a, b, e, i, j, npairs
    integer(kind=8) :: pow5(0:n)
    integer(kind=8) :: sum_ab, sum_cd, target
    integer, allocatable :: pairs(:,:)

    ! get x**5 
    do i = 0, n
        pow5(i) = int(i,8)**5
    end do

    !  a**5+b**5 pair for (a <= b)
    npairs = (n+1)*(n+2)/2
    allocate(pairs(2,npairs))
    j = 0
    do a = 0, n
        do b = a, n
            j = j + 1
            pairs(1,j) = a
            pairs(2,j) = b
        end do
    end do

    print *, "answer to a**5 + b**5 + c**5 +d**5 = e**5 for integer [a,b,c,d,e] in [1,200]"
    !print *, "solution"

    ! double loop of  (a,b) + (c,d)
    do i = 1, npairs
        sum_ab = pow5(pairs(1,i)) + pow5(pairs(2,i))
        do j = i, npairs   ! j start from i , cut half of the calculation cost
            sum_cd = pow5(pairs(1,j)) + pow5(pairs(2,j))
            target = sum_ab + sum_cd
            ! loop of e to examine if (a,b) + (c,d) == e**5
            do e = 0, n
            if (pow5(e) == target) then
                print *, " a=",pairs(1,i)," b=",pairs(2,i)," c=", pairs(1,j)," d=",pairs(2,j)," e=",e
            end if
            end do
        end do
    end do

    deallocate(pairs)
end program equation