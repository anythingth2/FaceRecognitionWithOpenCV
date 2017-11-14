import socket, cv2
import numpy as np
from io import StringIO

count = 0

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# can change port najaaaaa
# also disable firewall if on windows
server.bind(("0.0.0.0", 1234))

server.listen(1)
print('socker server initilized!')
while True:
  print('connecting...')
  conn, sock_addr = server.accept()
  print('connected')
  ultimate_buffer=b''
  while True:
    receiving_buffer = conn.recv(1024)
    print('receiving buffer')
    if not receiving_buffer: 
      break
    ultimate_buffer += receiving_buffer
    # print(receiving_buffer,type(receiving_buffer))
  # final image = numpy array of image
  print(ultimate_buffer.decode('utf-8'))
  # final_image = np.load(StringIO(ultimate_buffer))['frame']
  print('frame received')
  cv2.imwrite('tst' + str(count) + '.png', final_image)
  count += 1
  conn.close()