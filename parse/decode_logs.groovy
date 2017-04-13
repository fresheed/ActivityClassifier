import org.fresheed.actionlogger.utils.EventsLogCompressor
import org.fresheed.actionlogger.events.ActionEvent
import java.nio.file.Files

File bin_file=new File(args[0])
byte[] raw_bytes = Files.readAllBytes(bin_file.toPath())

compressor=new EventsLogCompressor()
List<ActionEvent> events=compressor.decompressEventsLog(raw_bytes, 3)

for (ActionEvent event: events){
    print "${event.getTimestamp()} "
    event.getValues().each {
       print "${it} "
    }      
    println ""
}