   implicit none
   integer,parameter:: SPLINE_INIT=-1,  SPLINE_NORMAL=1
   integer:: ny, ny1, j,tmp , Iflag
   real*8,dimension(:),allocatable:: yy, d,u,v,T, yy1,d1,u1,v1,T1
   print*, "interpolate 1d profiles for inlet (flow1d-inlet.dat)......"
   print*, "origin data : flow1d-inlet.dat.origin (y,d,u,v,T), new coordinate y1d.dat"   
   print*, "please input ny (orginate grid number),  ny1 (new) "
   read(*,*) ny, ny1 
   allocate(yy(ny),d(ny),u(ny),v(ny),T(ny), yy1(ny1),d1(ny1),u1(ny1),v1(ny1),T1(ny1))
   
   open(99,file="flow1d-inlet.dat.origin")
   read(99,*) 
   do j=1, ny 
    read(99,*) yy(j), d(j), u(j), v(j), T(j) 
   enddo
   close(99)
   
   open(100,file="y1d.dat")
   do j=1,ny1
   read(100,*) yy1(j) 
   enddo 
   
   print*, " input  1 for 6th Lagrange interpolation;   2 for 3rd Spline interpolation " 
   read(*,*) Iflag
   if(Iflag .eq. 1) then 
    print*, "Interpolation  (6th Lagrange) ..."
     do j=1,ny1
	 call inter1d_6th(yy1(j),d1(j),ny,yy,d)
	 call inter1d_6th(yy1(j),u1(j),ny,yy,u)   
 	 call inter1d_6th(yy1(j),v1(j),ny,yy,v)  
 	 call inter1d_6th(yy1(j),T1(j),ny,yy,T)
     enddo
   else 
    print*, "Interpolation  (3rd Spline) ..."
	!--------interpolation for d, u, v, T -------------
	call spline(ny,yy,d,0.d0,0.d0,SPLINE_INIT)
    do j=1,ny1
	 call spline(ny,yy,d,yy1(j),d1(j),SPLINE_NORMAL)
	enddo 
	
	call spline(ny,yy,u,0.d0,0.d0,SPLINE_INIT)
    do j=1,ny1
	 call spline(ny,yy,u,yy1(j),u1(j),SPLINE_NORMAL)
	enddo 	
	
	call spline(ny,yy,v,0.d0,0.d0,SPLINE_INIT)
    do j=1,ny1
	 call spline(ny,yy,v,yy1(j),v1(j),SPLINE_NORMAL)
	enddo 	
	
   	call spline(ny,yy,T,0.d0,0.d0,SPLINE_INIT)
    do j=1,ny1
	 call spline(ny,yy,T,yy1(j),T1(j),SPLINE_NORMAL)
	enddo 
  endif
!----------------------------------
   open(99,file="flow1d-inlet.dat")
   write(99,*) "variables=y, d, u, v, T" 
   do j=1,ny1 
    write(99,"(5F16.8)") yy1(j), d1(j), u1(j), v1(j), T1(j) 
   enddo
   close(99)
   

   open(99,file="flow-inlet.dat")
   do j=1,ny1 
    write(99,"(4F16.8)") d1(j), u1(j), v1(j), T1(j) 
   enddo
   close(99)
   
   deallocate(yy,d,u,v,T, yy1,d1,u1,v1,T1)
end 

!------------------------------------------------------------------------
! 6th order Langrage interpolation in one-dimension
 subroutine inter1d_6th(x0, f0, nx, xx,ff)
   implicit none
   integer:: nx,k,k0,ik,ikm,km,ka,kb
   real*8:: xx(nx),ff(nx), x0, f0, Ai(6)
   if(x0<=xx(1)) then
      f0=ff(1)
      return 
   endif

   if(x0>=xx(nx)) then
     f0=ff(nx)
     return
   endif

    k0=0
    do k=1,nx
      if(x0 >= xx(k)  .and. x0 < xx(k+1) ) then
	     k0=k
         goto 100
      endif
    enddo
 100  continue
    
	    ka=  max(4-k0,1)
        kb= min(nx+3-k0,6)
        f0=0.d0

	do k=ka,kb
	  ik=k0+k-3
          Ai(k)=1.d0
	   do km=ka,kb
	     ikm=k0+km-3
	     if(km .ne. k)      Ai(k)=Ai(k)*(x0-xx(ikm))/(xx(ik)-xx(ikm))
	    enddo
           f0=f0+Ai(k)*ff(ik)
	  enddo
     end



! Spline interpolation
! eg. 
! step1:  call spline (n, xx,yy, x0, y0, SPLINE_INIT) 
! Step2:  call spline (n, xx, yy, x0, y0, SPLINE_NORMAL)

 subroutine spline(n,xx,yy,x0,y0,Iflag)
 implicit none
 integer,parameter:: SPLINE_INIT=-1,  SPLINE_NORMAL=1
 integer,parameter:: Nmax=1000000
 integer:: n,j,k, Iflag
 integer,save:: Flag1=0
 real*8:: xx(n), yy(n), x0,y0,hj
 real*8:: a(n),b(n),c(n),d(n),h(n)
 real*8,save::  MM(Nmax)
 if(Iflag == SPLINE_INIT) then
   Flag1=1
   if(n > Nmax) then 
     print*, "Error !!! n> Nmax, please modify Nmax in subroutine spline() "
	 stop
   endif
   do j=2,n
    h(j)=xx(j)-xx(j-1)
   enddo
   do j=2,n-1
    a(j)=h(j)/(h(j)+h(j+1))
    b(j)=2.d0
    c(j)=h(j+1)/(h(j)+h(j+1))
    d(j)=6.d0*((yy(j+1)-yy(j))/h(j+1)-(yy(j)-yy(j-1))/h(j))/(h(j)+h(j+1))
   enddo
    a(1)=0.d0 ;  b(1)=1.d0 ;  c(1)= 0.d0 ;  d(1)=0.d0
	a(n)=0.d0 ;  b(n)=1.d0 ;  c(n)=0.d0;   d(n)=0.d0
   call LU3(n,a,b,c,d,MM)
  
   
  else
    if(Flag1 .ne. 1) then
	  print*, "Error !  Spline interpolation is not initialized !, please run spline(,,,,,-1) first"
	  stop
	 endif
!	 call find_k(n,xx,x0,k)         
	 call find_k_dichotomy(n,xx,x0,k)         
     j=k+1
	 hj=xx(j)-xx(j-1)
	 y0=(MM(j-1)*(xx(j)-x0)**3/6.d0+MM(j)*(x0-xx(j-1))**3/6.d0 &
	    +(yy(j-1)-MM(j-1)*hj*hj/6.d0)*(xx(j)-x0) + (yy(j)-MM(j)*hj*hj/6.d0)*(x0-xx(j-1)) ) /hj

	 
  endif
  end

!--------------搜索插值区间：  线性搜索法 ---------------------------------
  subroutine find_k(n,xx,x0,k)
  implicit none
  integer:: n,k,j
  real*8 :: xx(n), x0

   if(x0 <= xx(2)) then
     k=1
   else if(x0 >= xx(n-1)) then
     k=n-1
   else
    do j=2,n-2
	 if( x0 >= xx(j) .and. x0<= xx(j+1)) then
      k=j
      goto 100
     endif
    enddo
100  continue	
   endif

   end

!------------------------- 搜索插值区间： 二分法 ---------------------
subroutine find_k_dichotomy(n,xx,x0,k)
  implicit none
  integer:: n,k,kb,ke,k0
  real*8 :: xx(n), x0

   if(x0 <= xx(2)) then
     k=1
   else if(x0 >= xx(n-1)) then
     k=n-1
   else
     kb=2
	 ke=n-1
     do while( (ke-kb)>1) 
	   k0=(kb+ke)/2
       if(x0 <= xx(k0)) then 
          ke=k0 
       else
          kb=k0
       endif		  
	 enddo
     k=kb

   endif

   end

! Chasing method (3-line LU ) for 3-line Eq
  subroutine  LU3(N,a,b,c,d,x)
  implicit none
  integer:: N,j
  real*8,dimension(N):: a,b,c,d,x
  real*8,dimension(N):: AA,BB
  real*8:: tmp

   AA(1)=0.d0
   BB(1)=d(1)
   x(N)=d(N)
   x(1)=d(1)
  do j=2,N
    tmp=1.d0/(a(j)*AA(j-1)+b(j))
	AA(j)=-c(j)*tmp
	BB(j)=(d(j)-a(j)*BB(j-1))*tmp
  enddo
  do j=N-1,2,-1
   x(j)=AA(j)*x(j+1)+BB(j)
  enddo
  end



