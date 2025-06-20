package furhatos.app.faceskill

import org.zeromq.ZMQ

val context: ZMQ.Context = ZMQ.context(1)

fun getConnectedSocket(socketType: Int, port: String, receiveTimeout: Int = -1): ZMQ.Socket {
    val socket = context.socket(socketType)
    if (receiveTimeout >= 0) socket.receiveTimeOut = receiveTimeout

    if (socketType == ZMQ.SUB) {
        socket.subscribe("")
        socket.connect(port)
    }
    else {
        socket.bind("tcp://$port")
    }
    return socket
}
