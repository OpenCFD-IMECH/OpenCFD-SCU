    ! read-flow v1.2  for compression conner
    ! Mach 2.9-24 compression corner
    implicit doubleprecision (a-h,o-z) 
    real*8,allocatable,dimension(:,:,:):: d,u,v,w,T
    real*8,allocatable,dimension(:,:):: d0,u0,v0,w0,T0,us0,yh
    real*8,allocatable,dimension(:,:)::xx,yy
    real*8,allocatable,dimension(:):: pw0,uvd,up,Ret,Ut,length
    real*8 tt, length_min
    integer istep,if_read_average
    !------------------------------------------------------

    open(33,file="opencfd.in")
    do k=1,4
        read(33,*)
    enddo
    read(33,*) nx,ny,nz
    do k=1,7
        read(33,*)
    enddo
    read(33,*) tmp, tmp, SLZ
    read(33,*) 
    read(33,*) Re

    close(33)      
        
    !------------------------------------------------------ 
    
    allocate(d(nx,ny,nz),u(nx,ny,nz),v(nx,ny,nz),w(nx,ny,nz), T(nx,ny,nz))
    allocate(d0(nx,ny),u0(nx,ny),v0(nx,ny),w0(nx,ny),T0(nx,ny),us0(nx,ny),yh(nx,ny))
    allocate(xx(nx,ny),yy(nx,ny),pw0(nx),up(ny),uvd(ny),Ret(nx),Ut(nx),length(nx))    
    print*, "nx, ny ,nz =", nx,ny,nz
    print*, "SLZ, Re=", SLZ,Re
    !------------------------------------------------------
    
    gamma=1.4d0
    PI=3.14159265358979d0        
    
    seta0=24.d0*PI/180.d0   !conner angle
    hz=SLZ/nz               !展向周期边条
      
    T_ref=108.1d0
    Tsb=110.4/T_ref
    Tw=2.84d0
    Amu_w=1.0/Re*(1.0+Tsb)*sqrt(Tw**3)/(Tsb+Tw)

    !---------read grid message ---------------------------
    
    open(66,file="OCFD2d-Mesh.dat",form="unformatted")
    read(66) xx
    read(66) yy
    close(66)

    !---------read data file-------------------------------

    pw0=0.d0;d0=0.d0;u0=0.d0;v0=0.d0;w0=0.d0;T0=0.d0

    print*,"read opencfd.dat for (0) & read opencfd.dat.average for (1)"
    read(*,*) if_read_average
    if (if_read_average .eq. 1) then
        open(33,file="opencfd.dat.average",form="unformatted")
    else
        open(33,file="opencfd.dat",form="unformatted")
    end if
    
    print*, 'reading data......'

    read(33) Istep,tt
    print*, "Istep,tt=",Istep,tt    
    call read3d(33,d,nx,ny,nz)
    call read3d(33,u,nx,ny,nz)
    call read3d(33,v,nx,ny,nz)
    call read3d(33,w,nx,ny,nz)
    call read3d(33,T,nx,ny,nz)
    close(33)
    print*, "read data OK ..."
    
    !---------------------展项平均---------------------------------
    
        do i=1,nx
            do k=1,nz
                pw0(i)=pw0(i)+d(i,1,k)*T(i,1,k)
            enddo
        enddo

        do k=1,nz
            do j=1,ny
                do i=1,nx
                    d0(i,j)=d0(i,j)+d(i,j,k)
                    u0(i,j)=u0(i,j)+u(i,j,k)
                    v0(i,j)=v0(i,j)+v(i,j,k)
                    w0(i,j)=w0(i,j)+w(i,j,k)
                    T0(i,j)=T0(i,j)+T(i,j,k)
                enddo
            enddo
        enddo

    d0=d0/nz
    u0=u0/nz
    v0=v0/nz
    w0=w0/nz
    T0=T0/nz
    pw0=pw0/nz
    
    !------------------------------------------------------ 
 
    if (if_read_average .eq. 1) then
        open(44,file="pw-av.dat")
    else
        open(44,file="pw-ins.dat")
    endif
    write(44,*) "variables=x,p"
    do i=1,nx
        write(44,*) xx(i,1),pw0(i)
    enddo
    close(44)
    
    !------------------------------------------------------ 
    
!    do j=1,ny
!        do i=1,nx
!            if(xx(i,1) .le. 0) then
!                seta=0.d0
!            else
!                seta=seta0
!            endif
!            yh(i,j)=sqrt(   ( (xx(i,j)-xx(i,1)) * (1.d0-cos(seta)) ) **2     &
!            +((yy(i,j)-yy(i,1))*(1.d0-sin(seta))) **2  )
!            us0(i,j)=u0(i,j)*cos(seta)+v0(i,j)*sin(seta)    !切向速度
!        enddo
!    enddo

    !------------------------------------------------------       

    print*, "comput yh ..."
!   seta2d=0.d0
    yh=0.d0
    do j=1,ny
!        print*, "j=",j
        do i=1,nx!控制流场动点
            length_min=10000.d0
            do iw=1,nx!控制壁面动点，搜寻两点间最短距离length
                length(iw)=dsqrt((xx(i,j)-xx(iw,1))**2+(yy(i,j)-yy(iw,1))**2)
                if(length(iw).le.length_min)then
                    length_min=length(iw)
!                   i0=iw                    
                endif
            enddo
            yh(i,j)=length_min
!           seta2d(i,j)=seta(i0) 


            if(xx(i,1) .le. 0)then
                seta=0.d0
            else
                seta=seta0
            end if
            us0(i,j)=u0(i,j)*cos(seta)+v0(i,j)*sin(seta)    !切向速度           
        enddo
    enddo 
    
    !------------------------------------------------------    
    
    print*, "comput Cf ..."
    if(if_read_average .eq. 1)then
        open(55,file="cf-av.dat")
    else
        open(55,file="cf-ins.dat")    
    endif
    
    do i=1,nx
        us1=us0(i,2) ;  us2=us0(i,3) ; h1=yh(i,2) ; h2=yh(i,3)
        uy=(h2*h2*us1-h1*h1*us2)/(h2*h2*h1-h1*h1*h2)
        tw=Amu_w*uy
        Ut(i)=sqrt(abs(tw)/d0(i,1))
        Ret(i)= d0(i,1)*Ut(i)/Amu_w
        write(55,*) xx(i,1), 2.d0*tw, Ret(i)
    enddo
    close(55)
    !------------------------------------------------------  
    
    print*, "Comput delt0, delt1, delt2 "
    if (if_read_average .eq. 1) then    
        open(66,file="delta-av.dat")
    else
        open(66,file="delta-ins.dat")
    endif
    


    do i=1,nx
        delt0=0.d0
        delt1=0.d0
        delt2=0.d0
        do j=1,ny
            if( us0(i,j) .gt. 0.99d0) goto 200 
        enddo          
200     continue

        j0=j-1
        delt0= yh(i,j0) !边界层位移厚度
        ue=us0(i,j0)
        de=d0(i,j0)
        !print*, " j_0.99 , ue, de =", j0-1 , ue, de
a1 = j0 + 0.0
        do j=2,j0
            delt1=delt1+(1.d0-d0(i,j)*us0(i,j)/(ue*de))*(yh(i,j)-yh(i,j-1))
            delt2=delt2+d0(i,j)*us0(i,j)/(ue*de)*(1.d0-us0(i,j)/ue)*(yh(i,j)-yh(i,j-1))!动量边界层厚度
        enddo
        write(66,"(5f15.6)") a1,xx(i,1),delt0,delt1,delt2
    enddo
    close(66)
    
    !------------------------------------------------------

    print*, "output one-dimension profiles, please input i0 "
    read(*,*) i0
    print*, "x=",xx(i0,1)
    if (if_read_average .eq. 1) then 
        open(99,file="U1d-av.dat")
    else
        open(99,file="U1d-ins.dat")
    endif
    
    uvd(1)=0.d0
    up(1)=0.d0

    write(99,*) "variables=yp,up,uvd,u_log"
    do j=2,ny-1
        yp=yh(i0,j)*Ret(i0)
        up(j)=us0(i0,j)/ut(i0)
        uvd(j)=uvd(j-1)+sqrt(d0(i0,j)/d0(i0,1))*(up(j)-up(j-1))
        write(99,"(6f16.8)") yp, up(j),uvd(j),2.44*log(yp)+5.1d0
    enddo
    close(99)
    
    if (if_read_average .eq. 1) then 
        open(100,file="U1da-av.dat")
    else
        open(100,file="U1da-ins.dat")
    endif
    
    do j=1,ny
        write(100,"(6f16.8)") yh(i0,j),us0(i0,j),d0(i0,j),T0(i0,j)
    enddo
    close(100) 

    !------------------------------------------------------        

    if(if_read_average .eq. 1)then 
    print*,"writing flow2d_av..."
    open(44,file="flow2d-av.dat")
    write(44,*) "variables=x,y,d,u,v,w,T,p,s"
    write(44,*) "zone i= ",nx, " j= ",ny
    do j=1,ny
        do i=1,nx
            write(44,"(9f15.6)") xx(i,j),yy(i,j),d0(i,j),u0(i,j),v0(i,j),w0(i,j),T0(i,j),  &
            d0(i,j)*T0(i,j), T0(i,j)*d0(i,j)/d0(i,j)**gamma
        enddo
    enddo
    close(44)
    endif

    !------------------------------------------------------ 
    
    if(if_read_average .eq. 0)then    
    print*,"writing flow2d_xy..."
    print*, "please input k"
    read(*,*) k
    open(44,file="flow2d_xy.dat")
    write(44,*) "variables=x,y,z,d,u,v,w,T,p,s"
    write(44,*) "zone i= ",nx, " j= ",ny
!   k=nz/2+1
    z=(k-1.)*hz
    do j=1,ny
        do i=1,nx
            write(44,"(9f15.6)") xx(i,j),yy(i,j),z,d(i,j,k),u(i,j,k),v(i,j,k),w(i,j,k),T(i,j,k),  &
            d(i,j,k)*T(i,j,k),T(i,j,k)*d(i,j,k)/d(i,j,k)**gamma
        enddo
    enddo
    close(44)    
    
    !------------------------------------------------------     

    print*,"writing flow2d_xz..."
    print*, "please input j"
    read(*,*) j
    print*, "yn=", sqrt((xx(1,j)-xx(1,1))**2+(yy(1,j)-yy(1,1))**2)

    open(44,file="flow2d_xz.dat")
    write(44,*) "variables=x,y,z,d,u,v,w,T,p,s,us"
    write(44,*) "zone i= ",nx, " j= ",nz

    do k=1,nz
        do i=1,nx
            z=(k-1.)*hz
            if(xx(i,j) .gt. 0 ) then
                us=u(i,j,k)*cos(seta0)+v(i,j,k)*sin(seta0)
            else
                us=u(i,j,k)
            endif

            write(44,"(10f15.6)") xx(i,j),yy(i,j),z,d(i,j,k),u(i,j,k),   &
            v(i,j,k),w(i,j,k),T(i,j,k),   &
            d(i,j,k)*T(i,j,k), T(i,j,k)*d(i,j,k)/d(i,j,k)**gamma,us
        enddo
    enddo
    close(44)

    !------------------------------------------------------    
    
    print*,"writing flow2d_yz..."
    print*, "please input i"
    read(*,*) i
    print*, "x=", xx(i,1)
    
    open(44,file="flow2d_yz.dat")
    write(44,*) "variables=x,y,z,d,u,v,w,T,p,s"
    write(44,*) "zone i= ",ny, " j= ",nz

    do k=1,nz
        do j=1,ny
            z=(k-1.)*hz
            write(44,"(9f15.6)") xx(i,j),yy(i,j),z,d(i,j,k),u(i,j,k),   &
            v(i,j,k),w(i,j,k),T(i,j,k),   &
            d(i,j,k)*T(i,j,k), T(i,j,k)*d(i,j,k)/d(i,j,k)**gamma
        enddo
    enddo
    close(44)
    end if
    
    !------------------------------------------------------
    
    end

    subroutine read3d(no,u,nx,ny,nz)
    real*8 u(nx,ny,nz)
    print*, "read 3d data ..."
    do k=1,nz
        read(no) u(:,:,k)
    enddo
    end
