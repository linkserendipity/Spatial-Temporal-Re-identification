# coding=utf-8
import smtplib  
from email.mime.text import MIMEText  
from email.header import Header
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart  
import time
#邮件发送函数
#SMTPHost 邮件服务器
#fromAccount 发件邮箱
#fromPasswd 发件邮件授权码，注意并不是邮箱登录密码
#toAccount 收件邮箱
#subject 邮件标题
#content 邮件正文

def sendMail(content='what content?', SMTPHost='smtp.126.com', fromAccount='testgpu318@126.com', fromPasswd='YXSVONCFBRKATTTP', toAccount='testgpu318@126.com', subject='eval_acc'):    

    #构建邮件
    msg = MIMEMultipart()
    msg['Subject'] = Header(subject, 'utf-8')
    msg['From'] = fromAccount
    msg['To'] = toAccount

    # content = '<html><body><h3>主题内容</h3>' + '<p><a href="www.weatherfood.com">谭某人的博客</a></p>' + '</body></html>'
    puretext = MIMEText(content, 'html', 'utf-8')

    msg.attach(puretext)

    #! 附件
    # xlsxpart = MIMEApplication(open('tank/mask.txt', 'rb').read()) #这里填写你自己目录下的附件文件
    # xlsxpart.add_header('Content-Disposition', 'attachment', filename='tank/mask.txt')
    # msg.attach(xlsxpart)

    #使用smtplib模块发送邮件
    email_client = smtplib.SMTP(SMTPHost)
    # email_client.connect(SMTPHost, 25)
    email_client.login(fromAccount, fromPasswd)
    email_client.sendmail(fromAccount, toAccount, msg.as_string())
    email_client.quit()
    print('mail success')            

if __name__ == '__main__':
    acc=0.85
    sendMail(content='result={}'.format(acc), toAccount='m18679060131@163.com',subject='t e s t')
            # num1=0
            # while True:
            #     try:
            #         send_mail("auto_new.py:acc={}".format(acc))
            #         send_mail("auto_new.py:acc={}".format(acc), toAccount='testgpuinfo888@126.com')
            #         break
            #     except:
            #         print('WWarning: email not sent.')
            #         num1+=1
            #         if num1>10:
            #             print('mail failed')
            #             break      