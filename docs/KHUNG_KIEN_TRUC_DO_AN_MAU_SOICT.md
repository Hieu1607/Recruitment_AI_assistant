# Khung kien truc do an mau SoICT

## Muc dich

Tai lieu nay chi tach **khung cau truc** cua file `SoICT_Do_an_tot_nghiep_Ung_dung_Nguyen_Tung_Lam.pdf`, khong lay noi dung chuyen mon cua de tai mau. Muc tieu la de ban co mot bo khung ro rang, co the thay the noi dung bang de tai that cua minh.

Quy uoc:

- Cac cum trong `[ngoac vuong]` la cho de ban thay bang noi dung thuc te.
- Cac ten muc mang tinh dac thu cua de tai mau da duoc tru tuong hoa thanh nhom chuc nang, use case, giai phap, dong gop.
- So thu tu chuong/muc duoc giu theo dung logic cua mau.

## 1. Kien truc tong the cua tai lieu

Do an mau duoc to chuc thanh 5 khoi lon:

1. **Khoi nhan dien tai lieu**
   - Trang bia.
2. **Khoi mo dau**
   - Loi cam on.
   - Tom tat tieng Viet.
   - Abstract tieng Anh.
3. **Khoi dieu huong**
   - Muc luc.
   - Danh muc hinh ve.
   - Danh muc bang bieu.
   - Danh muc thuat ngu va tu viet tat.
4. **Khoi noi dung chinh**
   - 6 chuong noi dung, di tu gioi thieu -> phan tich yeu cau -> cong nghe -> thiet ke/trien khai/danh gia -> dong gop noi bat -> ket luan.
5. **Khoi ket tai lieu**
   - Tai lieu tham khao.
   - Phu luc.

## 2. Ty trong noi dung cua mau

Phan bo cua mau cho thay trong tam nam o thiet ke he thong, trien khai va phan dong gop ky thuat:

| Phan | Trang bat dau trong noi dung chinh | Vai tro |
| --- | --- | --- |
| Chuong 1 | 1 | Gioi thieu de tai, van de, muc tieu, pham vi, bo cuc |
| Chuong 2 | 5 | Khao sat hien trang, phan tich chuc nang, dac ta use case, yeu cau phi chuc nang |
| Chuong 3 | 36 | Giai thich cong nghe/thu vien/dich vu duoc chon |
| Chuong 4 | 48 | Thiet ke kien truc, thiet ke chi tiet, xay dung, kiem thu, trien khai |
| Chuong 5 | 97 | Cac giai phap ky thuat va dong gop noi bat |
| Chuong 6 | 121 | Ket luan va huong phat trien |
| Tai lieu tham khao | 124 | Khoa tai lieu |
| Phu luc | 125 | Tai lieu bo tro |

Neu tinh theo logic cua mau thi:

- Chuong 4 la chuong trong tam lon nhat.
- Chuong 5 la chuong "gia tri gia tang", dung de noi ro cac dong gop ky thuat thay vi chi mo ta tinh nang.
- Chuong 2 di rat sau vao phan tich yeu cau va use case.

## 3. Cay muc chi tiet da tru tuong hoa

### 3.1. Khoi dau tai lieu

1. **Trang bia**
   - [Ten truong]
   - [Loai tai lieu: Do an tot nghiep / Khoa luan / Luan van]
   - [Ten de tai]
   - [Ten sinh vien]
   - [Email / MSSV neu can]
2. **Loi cam on**
3. **Tom tat noi dung do an**
4. **Abstract**
5. **Muc luc**
6. **Danh muc hinh ve**
7. **Danh muc bang bieu**
8. **Danh muc thuat ngu va tu viet tat**

### 3.2. Chuong 1 - Gioi thieu de tai

Khung cua chuong mo dau rat ro va gon:

1. `1.1 Dat van de`
   - Vi sao bai toan xuat hien.
   - Nhu cau thuc te hoac khoang trong hien tai.
2. `1.2 Muc tieu va pham vi de tai`
   - Muc tieu chinh.
   - Gioi han bai toan.
   - Doi tuong ap dung.
3. `1.3 Dinh huong giai phap`
   - Huong tiep can tong quat.
   - Hinh dung so bo ve he thong/giai phap se xay.
4. `1.4 Bo cuc do an`
   - Tom tat moi chuong se noi gi.

### 3.3. Chuong 2 - Khao sat va phan tich yeu cau

Day la chuong phan tich nghiep vu va yeu cau. Mau di theo 4 lop:

1. `2.1 Khao sat hien trang`
   - So sanh san pham/nen tang lien quan.
   - Rut ra diem manh, diem yeu, khoang trong.
2. `2.2 Tong quan chuc nang`
   - `2.2.1 Bieu do use case tong quat`
   - `2.2.2` den `2.2.12`: moi muc la **mot bieu do use case phan ra cho tung nhom chuc nang**.
   - Khung nay cho thay mau uu tien phan ra chuc nang theo tung phan he thay vi mo ta chung chung.
3. `2.3 Dac ta chuc nang`
   - `2.3.1` den `2.3.7`: moi muc la **mot use case trong tam** duoc dac ta chi tiet.
   - Trong mau, moi use case dac ta theo form gan nhu chuan:
     - Ma use case
     - Ten use case
     - Tac nhan
     - Tien dieu kien
     - Hau dieu kien (neu co)
     - Luong su kien chinh
     - Luong thay the / ngoai le (neu co)
4. `2.4 Yeu cau phi chuc nang`
   - `2.4.1 Hieu nang`
   - `2.4.2 Do tin cay`
   - `2.4.3 Kha nang su dung`
   - `2.4.4 Bao mat`
   - `2.4.5 Tinh de bao tri`

### 3.4. Chuong 3 - Cong nghe su dung

Chuong nay cua mau khong viet theo van phong "tong quan hoc thuat", ma viet theo logic chon cong nghe de xay he thong. Khung cua no la:

1. `3.1 [Thu vien / nen tang cot loi 1]`
2. `3.2 [Thu vien / nen tang cot loi 2]`
3. `3.3 [API / dich vu ngoai / AI service neu co]`
4. `3.4 Cong nghe phat trien frontend`
   - `3.4.1 [Ngon ngu / runtime]`
   - `3.4.2 [Framework / thu vien giao dien]`
5. `3.5 Cong nghe phat trien backend`
   - `3.5.1 [Ngon ngu]`
   - `3.5.2 [Framework]`
6. `3.6 Co so du lieu`
   - `3.6.1 [CSDL chinh]`
   - `3.6.2 [CSDL phu / vector / cache / search]`
7. `3.7 [Dich vu luu tru tep/media]`
8. `3.8 [Docker / cong cu dong goi / van hanh]`

Moi muc cong nghe trong mau thuong tra loi 3 cau hoi:

- Cong nghe nay la gi?
- Vi sao chon no?
- No duoc ap dung vao thanh phan nao cua do an?

### 3.5. Chuong 4 - Thiet ke, trien khai va danh gia he thong

Day la chuong "xuong song" cua toan bo do an mau. Cau truc cua chuong nay rat manh va rat nen hoc theo:

1. `4.1 Thiet ke kien truc`
   - `4.1.1 Lua chon kien truc phan mem`
   - `4.1.2 Thiet ke tong quan`
   - `4.1.3 Thiet ke chi tiet goi`
2. `4.2 Thiet ke chi tiet`
   - `4.2.1 Thiet ke giao dien`
   - `4.2.2 Thiet ke lop`
   - `4.2.3 Thiet ke co so du lieu`
3. `4.3 Xay dung ung dung`
   - `4.3.1 Thu vien va cong cu su dung`
   - `4.3.2 Ket qua dat duoc`
   - `4.3.3 Minh hoa cac chuc nang chinh`
4. `4.4 Kiem thu`
   - `4.4.1 Kiem thu tuong thich`
   - `4.4.2 Kiem thu hop den`
5. `4.5 Trien khai`
   - `4.5.1 Moi truong phat trien`
   - `4.5.2 Trien khai thu nghiem`

Y nghia cua khung nay:

- `4.1` giai quyet cau hoi "he thong duoc to chuc nhu the nao".
- `4.2` giai quyet cau hoi "moi thanh phan quan trong duoc thiet ke chi tiet ra sao".
- `4.3` giai quyet cau hoi "da xay duoc gi, minh hoa the nao".
- `4.4` giai quyet cau hoi "da kiem chung chat luong bang cach nao".
- `4.5` giai quyet cau hoi "he thong duoc dat vao moi truong thuc te ra sao".

### 3.6. Chuong 5 - Cac giai phap va dong gop noi bat

Chuong nay la diem rat hay cua mau. No tach rieng khoi chuong 4 de khong lam loang phan "dong gop". Thay vi chi liet ke tinh nang, mau trinh bay cac **bai toan ky thuat** va **cach giai quyet**.

Mau co 7 cum dong gop, va moi cum gan nhu la mot mini-case-study:

1. `5.1 [Dong gop / bai toan ky thuat 1]`
   - `5.1.1 Dat van de`
   - `5.1.2 Giai phap`
   - `5.1.3 Ket qua dat duoc`
2. `5.2 [Dong gop / bai toan ky thuat 2]`
   - `5.2.1 Dat van de`
   - `5.2.2 Giai phap`
3. `5.3 [Dong gop / bai toan ky thuat 3]`
   - `5.3.1 Dat van de`
   - `5.3.2 Giai phap`
   - `5.3.3 Ket qua dat duoc`
4. `5.4 [Dong gop / bai toan ky thuat 4]`
   - `5.4.1 Dat van de`
   - `5.4.2 Giai phap`
5. `5.5 Thiet ke he thong`
   - `5.5.1 Dat van de`
   - `5.5.2 Giai phap`
   - `5.5.3 Ket qua dat duoc`
6. `5.6 [Dong gop / bai toan ky thuat 6]`
   - `5.6.1 Dat van de`
   - `5.6.2 Giai phap`
   - `5.6.3 Ket qua dat duoc`
7. `5.7 [Dong gop / bai toan ky thuat 7]`
   - `5.7.1 Dat van de`
   - `5.7.2 Giai phap`
   - `5.7.3 Ket qua dat duoc`

Khung nay rat hop neu do an cua ban co nhieu quyet dinh ky thuat de ke thanh "dong gop". Neu de tai cua ban it dong gop, van co the giu chuong 5 nhung giam so cum xuong 3-5 muc.

### 3.7. Chuong 6 - Ket luan va huong phat trien

Khung nay ngan, dut khoat:

1. `6.1 Ket luan`
   - Tong ket nhung gi da dat duoc.
   - Doi chieu muc tieu ban dau.
2. `6.2 Huong phat trien`
   - Gioi han con ton tai.
   - Huong mo rong tiep theo.

### 3.8. Phan cuoi tai lieu

1. **Tai lieu tham khao**
2. **Phu luc**

## 4. Kieu noi dung duoc lap lai trong mau

Neu muon "ap dung dung tinh than cua mau", khong chi giu ten chuong ma con nen giu cac loai tai lieu xuat hien trong tung chuong:

| Khu vuc | Kieu noi dung xuat hien trong mau |
| --- | --- |
| Chuong 2 | Bang so sanh, bieu do use case tong quat, bieu do use case phan ra, bang dac ta use case |
| Chuong 3 | Mo ta cong nghe, uu diem/han che, ly do lua chon, vai tro trong he thong |
| Chuong 4 | Bieu do kien truc, bieu do goi/module, wireframe, bieu do lop, bieu do trinh tu, so do CSDL, bang mo ta bang/truong, anh giao dien, bang kiem thu, bang moi truong trien khai |
| Chuong 5 | Bai toan -> giai phap -> ket qua; hinh minh hoa, so do module, phan tich quyet dinh ky thuat |
| Chuong 6 | Tong ket va de xuat phat trien |

## 5. Khung suon co the dung lai ngay

Ban co the copy khung duoi day va thay the dan:

```md
# [TEN DO AN]

## Loi cam on

## Tom tat noi dung do an

## Abstract

## Muc luc

## Danh muc hinh ve

## Danh muc bang bieu

## Danh muc thuat ngu va tu viet tat

# CHUONG 1. GIOI THIEU DE TAI
## 1.1 Dat van de
## 1.2 Muc tieu va pham vi de tai
## 1.3 Dinh huong giai phap
## 1.4 Bo cuc do an

# CHUONG 2. KHAO SAT VA PHAN TICH YEU CAU
## 2.1 Khao sat hien trang
## 2.2 Tong quan chuc nang
### 2.2.1 Bieu do use case tong quat
### 2.2.2 Bieu do use case phan ra [Nhom chuc nang 1]
### 2.2.3 Bieu do use case phan ra [Nhom chuc nang 2]
### 2.2.4 Bieu do use case phan ra [Nhom chuc nang 3]
### 2.2.n Bieu do use case phan ra [Nhom chuc nang n]
## 2.3 Dac ta chuc nang
### 2.3.1 Dac ta use case [Use case trong tam 1]
### 2.3.2 Dac ta use case [Use case trong tam 2]
### 2.3.n Dac ta use case [Use case trong tam n]
## 2.4 Yeu cau phi chuc nang
### 2.4.1 Hieu nang
### 2.4.2 Do tin cay
### 2.4.3 Kha nang su dung
### 2.4.4 Bao mat
### 2.4.5 Tinh de bao tri

# CHUONG 3. CONG NGHE SU DUNG
## 3.1 [Thu vien / nen tang cot loi 1]
## 3.2 [Thu vien / nen tang cot loi 2]
## 3.3 [API / dich vu ngoai]
## 3.4 Cong nghe phat trien frontend
### 3.4.1 [Ngon ngu]
### 3.4.2 [Framework]
## 3.5 Cong nghe phat trien backend
### 3.5.1 [Ngon ngu]
### 3.5.2 [Framework]
## 3.6 Co so du lieu
### 3.6.1 [CSDL chinh]
### 3.6.2 [CSDL phu]
## 3.7 [Dich vu luu tru]
## 3.8 [Docker / cong cu van hanh]

# CHUONG 4. THIET KE, TRIEN KHAI VA DANH GIA HE THONG
## 4.1 Thiet ke kien truc
### 4.1.1 Lua chon kien truc phan mem
### 4.1.2 Thiet ke tong quan
### 4.1.3 Thiet ke chi tiet goi
## 4.2 Thiet ke chi tiet
### 4.2.1 Thiet ke giao dien
### 4.2.2 Thiet ke lop
### 4.2.3 Thiet ke co so du lieu
## 4.3 Xay dung ung dung
### 4.3.1 Thu vien va cong cu su dung
### 4.3.2 Ket qua dat duoc
### 4.3.3 Minh hoa cac chuc nang chinh
## 4.4 Kiem thu
### 4.4.1 Kiem thu tuong thich
### 4.4.2 Kiem thu hop den
## 4.5 Trien khai
### 4.5.1 Moi truong phat trien
### 4.5.2 Trien khai thu nghiem

# CHUONG 5. CAC GIAI PHAP VA DONG GOP NOI BAT
## 5.1 [Dong gop 1]
### 5.1.1 Dat van de
### 5.1.2 Giai phap
### 5.1.3 Ket qua dat duoc
## 5.2 [Dong gop 2]
### 5.2.1 Dat van de
### 5.2.2 Giai phap
## 5.3 [Dong gop 3]
### 5.3.1 Dat van de
### 5.3.2 Giai phap
### 5.3.3 Ket qua dat duoc
## 5.n [Dong gop n]
### 5.n.1 Dat van de
### 5.n.2 Giai phap
### 5.n.3 Ket qua dat duoc

# CHUONG 6. KET LUAN VA HUONG PHAT TRIEN
## 6.1 Ket luan
## 6.2 Huong phat trien

# TAI LIEU THAM KHAO

# PHU LUC
```

## 6. Cach ap khung nay vao do an cua ban

Neu ban muon giu "phom" cua mau nhung phu hop voi de tai cua minh, cach ap dung hop ly nhat la:

1. Giu nguyen khoi dau, khoi dieu huong, 6 chuong chinh, tai lieu tham khao va phu luc.
2. Giu nguyen logic Chuong 1, 4, 6 vi day la phan rat chung va de dung lai.
3. O Chuong 2, thay ten cac nhom chuc nang va use case theo de tai cua ban, nhung van nen giu cau truc:
   - Khao sat hien trang
   - Tong quan chuc nang
   - Dac ta use case trong tam
   - Yeu cau phi chuc nang
4. O Chuong 3, thay cong nghe cu the nhung van giu cach nhom theo frontend/backend/database/service.
5. O Chuong 5, chi giu cac dong gop thuc su co gia tri ky thuat; khong bien chuong nay thanh ban sao cua Chuong 4.

## 7. Nhan xet quan trong ve mau

Diem manh cua bo khung nay nam o 3 y:

1. **Tac biet ro "yeu cau" va "giai phap"**: Chuong 2 dung de phan tich bai toan, Chuong 4 va 5 moi dung de noi cach giai.
2. **Co chuong rieng cho dong gop ky thuat**: Dieu nay giup do an trong chat hon rat nhieu.
3. **Day du truc tham chieu ky thuat**: use case, kien truc, lop, CSDL, kiem thu, trien khai, phu luc.

Neu ban muon, buoc tiep theo toi co the tao them mot file thu hai: **khung da ca nhan hoa cho chinh do an cua ban**, tuc la lay bo khung tren va doi ten cac muc thanh dung voi de tai hien tai cua ban.
