import difflib
compnames = ['SLSU','AXIU', 'CXDU', 'TCLU', 'CBHU', 'TEMU', 'SEGU', 'YMLU', 'DFSU', 'MEDU', 'UACU', 'MRKU', 'FCIU', 'BSIU', 'SUDU', 'TCNU', 'BMOU', 'WDFU', 'PHRU', 'CMAU', 'APHU', 'TCKU', 'EISU', 'NEXU', 'EITU', 'TRIU', 'PONU', 'BEAU', 'FSCU', 'KMBU', 'WWWU', 'MOTU', 'TRLU', 'IRSU', 'TRHU', 'TGHU', 'ZCSU', 'MSKU', 'WKCU', 'CRSU', 'DRYU', 'SKHU', 'STXU', 'TWCU', 'SNWU', 'CLHU', 'ECMU', 'HALU', 'MSCU', 'UESU', 'MAGU', 'CAIU', 'TTNU', 'GESU', 'IPXU', 'HDMU', 'CARU', 'HMCU', 'FESU', 'APZU', 'PCIU', 'GATU', 'UETU', 'GLDU', 'NLLU', 'HLXU', 'HLBU', 'MAEU', 'GCSU', 'XINU', 'IMTU', 'POLU', 'OOLU', 'SBAU', 'TOLU', 'NIUU', 'YMMU', 'EGHU', 'MNBU', 'MSLU', 'ZIMU', 'MRSU', 'DAYU', 'LYGU', 'EGSU', 'CAXU', 'WFHU', 'HJCU', 'RFCU', 'HNSU', 'CZTU', 'TLXU', 'GRXU', 'TIHU', 'CNSU', 'RMKU', 'VARU', 'SKLU', 'KKTU', 'SKIU', 'PARU', 'CCLU', 'SFFU', 'MOEU', 'HASU', 'FRLU', 'MMAU', 'ESPU', 'MBFU', 'BSBU', 'CRXU', 'SIMU', 'HJLU', 'AMFU', 'SGRU', 'KKFU', 'TLLU', 'MSWU', 'FCGU', 'NYKU', 'STTU', 'RWLU', 'TGBU', 'YMKU', 'CSLU', 'WHLU', 'MTEU', 'OOCU', 'EMCU', 'NIDU', 'ETMU', 'SNBU', 'SZLU', 'CXRU', 'ONTU', 'SSXU', 'DVDU', 'TSTU', 'KMTU', 'TGCU', 'GURU', 'INBU', 'MBOU', 'IRNU', 'TYLU', 'PRSU', 'OCGU', 'ESGU', 'CPSU', 'SITU', 'IEMU', 'UAGU', 'SCTU', 'TGAU', 'APRU', 'MWMU', 'SSNU', 'CSDU', 'WSDU', 'UNIU', 'TSLU', 'MWCU', 'IKSU', 'GVCU']
def checkCorrect(text):
        maxratio= 0
        temp = text
        for compname in compnames: 
                        #print(prenumber[:4], compname)                  
                        ratio = difflib.SequenceMatcher(None, text, compname).quick_ratio()
                        if maxratio > ratio:
                                maxratio = maxratio 
                        else:
                                maxratio = ratio
                                temp = compname
                        #print(maxratio)
                        #print(ratio)
        return maxratio, temp

def revise(prenumber):
        # 前四位变化

        if '0' or '1' or '7' in prenumber[:5]:
                        p1 = prenumber[:4].replace('1','I')
                        p1 = p1[:4].replace('7','I')
                        p1 = p1[:4].replace('0','O')
                        p2 = prenumber[4:]
                        prenumber = p1+p2
                        #print(prenumber)
        
        pre_str = prenumber[:4]

        correct_prop, out = checkCorrect(pre_str)
        #print("T",correct_prop)

        if correct_prop != 1:
                new_pre_str = pre_str
                new_pre_str = new_pre_str.replace("O","D")
                new_pre_str = new_pre_str.replace("I","T")
                new_pre_str = new_pre_str.replace("W","V") 
                new_correct_prop, out  = checkCorrect(new_pre_str)
                #print(new_correct_prop,out)
                if new_correct_prop == 1:
                        pre_str = new_pre_str

            

        prenumber = prenumber.replace(prenumber[:4], pre_str)
        # 最后一位变 ‘1’ 
        if len(prenumber) >=15:

                prenumber = prenumber[:15]
                #print('1')
                if prenumber[14] != '1' and prenumber[14] != '0' and prenumber[14] != '2':
                        prenumber = prenumber[:14] + '1'  
        if len(prenumber) < 15 and len(prenumber) > 1 and prenumber[-1] == 'I' :
                prenumber = prenumber[:-1] + '1'
        # 中间位变化
        if len(prenumber) >= 11:
                p2 = prenumber[4:11].replace('O', '0')
                p2 = p2.replace('I', '1')
                p2 = p2.replace('I', '1')
                p2 = p2.replace('K', '5')
                p2 = p2.replace('G', '6')
                p2 = p2.replace('C', '5')
                p2 = p2.replace('L', '1')
                p1 = prenumber[:4]
                p3 = prenumber[11:]
                prenumber = p1+p2+p3
        
        # 前四位英文U对准    STU905083845G1  TSLU905083845G1
        numlist = '0123456789'
        englist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        # 针对前三位英文 缺少‘U’
        if len(prenumber) >= 10 and prenumber[2]!='U' and prenumber[3] in numlist and prenumber[2] in englist:
                prenumber = prenumber[:3] + 'U' + prenumber[3:]
        # 针对‘U’前缺少英文
        if len(prenumber) >= 10 and prenumber[2]=='U' and prenumber[3] in numlist and prenumber[2] in englist:
                pre_str = prenumber[:3]
                correct_prop, pre_str = checkCorrect(pre_str)
                prenumber = pre_str + prenumber[3:]
        

        if len(prenumber) == 14:
                prenumber = prenumber + '1'
     
        return  prenumber
# a = 'EDU628091422G1'
# print(a[3])
# print(revise(a))