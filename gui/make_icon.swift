import AppKit

let size = NSSize(width: 1024, height: 1024)
let image = NSImage(size: size)
image.lockFocus()

let background = NSBezierPath(roundedRect: NSRect(x: 32, y: 32, width: 960, height: 960), xRadius: 220, yRadius: 220)
let gradient = NSGradient(colors: [
    NSColor(calibratedRed: 0.12, green: 0.20, blue: 0.48, alpha: 1),
    NSColor(calibratedRed: 0.30, green: 0.12, blue: 0.58, alpha: 1)
])!
gradient.draw(in: background, angle: -45)

NSColor.white.withAlphaComponent(0.13).setFill()
NSBezierPath(ovalIn: NSRect(x: 530, y: 500, width: 510, height: 510)).fill()

let page = NSBezierPath(roundedRect: NSRect(x: 220, y: 175, width: 584, height: 674), xRadius: 56, yRadius: 56)
NSColor.white.setFill()
page.fill()

let fold = NSBezierPath()
fold.move(to: NSPoint(x: 650, y: 849))
fold.line(to: NSPoint(x: 804, y: 695))
fold.line(to: NSPoint(x: 650, y: 695))
fold.close()
NSColor(calibratedWhite: 0.86, alpha: 1).setFill()
fold.fill()

let ink = NSColor(calibratedRed: 0.16, green: 0.18, blue: 0.35, alpha: 1)
let paragraph = NSMutableParagraphStyle()
paragraph.alignment = .center
let attributes: [NSAttributedString.Key: Any] = [
    .font: NSFont.systemFont(ofSize: 210, weight: .bold),
    .foregroundColor: ink,
    .paragraphStyle: paragraph
]
("∑" as NSString).draw(in: NSRect(x: 250, y: 395, width: 524, height: 260), withAttributes: attributes)

ink.withAlphaComponent(0.72).setFill()
for y in [330.0, 270.0] {
    NSBezierPath(roundedRect: NSRect(x: 310, y: y, width: 404, height: 22), xRadius: 11, yRadius: 11).fill()
}

image.unlockFocus()
guard let tiff = image.tiffRepresentation,
      let bitmap = NSBitmapImageRep(data: tiff),
      let png = bitmap.representation(using: .png, properties: [:]) else { exit(1) }
let outputPath = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : (URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent().appendingPathComponent("AppIcon-1024.png").path)
try png.write(to: URL(fileURLWithPath: outputPath))
