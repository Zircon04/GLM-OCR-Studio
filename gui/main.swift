import AppKit
import Darwin
import Foundation
import ImageIO
import PDFKit
import WebKit

private let installDir: String = {
    if let env = ProcessInfo.processInfo.environment["GLM_OCR_DIR"], FileManager.default.fileExists(atPath: env) {
        return env
    }
    let exeURL = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath()
    var dir = exeURL.deletingLastPathComponent()
    if dir.lastPathComponent == "MacOS" && dir.deletingLastPathComponent().lastPathComponent == "Contents" {
        let appParent = dir.deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        if FileManager.default.fileExists(atPath: appParent.appendingPathComponent("manager/job_runner.py").path) {
            return appParent.path
        }
    }
    if dir.lastPathComponent == "gui" {
        let parent = dir.deletingLastPathComponent()
        if FileManager.default.fileExists(atPath: parent.appendingPathComponent("manager/job_runner.py").path) {
            return parent.path
        }
    }
    let cwd = FileManager.default.currentDirectoryPath
    if FileManager.default.fileExists(atPath: URL(fileURLWithPath: cwd).appendingPathComponent("manager/job_runner.py").path) {
        return cwd
    }
    let defaultPath = "/Users/liqtranq/Documents/GLM-OCR-local"
    if FileManager.default.fileExists(atPath: defaultPath) {
        return defaultPath
    }
    return dir.path
}()
private let startScript = installDir + "/start-glm-ocr.command"
private let stopScript = installDir + "/stop-glm-ocr.command"
private let sdkPython = installDir + "/.venv-sdk/bin/python"
private let jobRunner = installDir + "/manager/job_runner.py"
private let resultInfoScript = installDir + "/manager/result_info.py"
private let libraryScript = installDir + "/manager/library_index.py"
private let cleanupScript = installDir + "/manager/cleanup_manager.py"
private let resultsDir = installDir + "/results"
private let outputDirKey = "GLMOutputDirectory"
private let logFile = installDir + "/logs/mlx-vlm.log"
private let profileFile = installDir + "/config/performance-profile.env"
private let queueFile = installDir + "/manager/queue-state.json"

private struct PerformanceProfile {
    let key: String
    let title: String
    let serverSequences: Int
    let sdkWorkers: Int
    let pageQueue: Int
    let regionQueue: Int
    let reservedGB: Int
}

private struct QueueItem: Codable {
    var id: UUID
    var source: String
    var output: String
    var status: String
    var progress: Double
    var pagesDone: Int
    var pagesTotal: Int
    var etaSeconds: Int?
    var message: String
    var addedAt: Date

    var name: String { URL(fileURLWithPath: source).lastPathComponent }
}

    private struct LibraryEntry {
    let title: String
    let path: String
    let source: String
    let pages: Int
        let snippet: String
}

final class AppDelegate: NSObject, NSApplicationDelegate, NSWindowDelegate, NSTableViewDataSource, NSTableViewDelegate {
    private var window: NSWindow!
    private var tabs: NSTabView!
    private var queueTable: NSTableView!
    private var libraryTable: NSTableView!
    private var logView: NSTextView!
    private var structureView: NSTextView!
    private var previewWeb: WKWebView!
    private var originalImage: NSImageView!
    private var enhancedImage: NSImageView!
    private var originalCaption: NSTextField!
    private var enhancedCaption: NSTextField!
    private var imagePopup: NSPopUpButton!
    private var compareSlider: NSSlider!
    private var serverLabel: NSTextField!
    private var serverDot: NSImageView!
    private var profilePopup: NSPopUpButton!
    private var profileDescription: NSTextField!
    private var progressLabel: NSTextField!
    private var detailLabel: NSTextField!
    private var memoryLabel: NSTextField!
    private var progressBar: NSProgressIndicator!
    private var runButton: NSButton!
    private var pauseButton: NSButton!
    private var startButton: NSButton!
    private var stopButton: NSButton!
    private var upButton: NSButton!
    private var downButton: NSButton!
    private var removeButton: NSButton!
    private var retryButton: NSButton!
    private var cleanupButton: NSButton!
    private var enhanceCheckbox: NSButton!
    private var enhancementModePopup: NSPopUpButton!
    private var fullMDCheckbox: NSButton!
    private var splitMDCheckbox: NSButton!
    private var searchablePDFCheckbox: NSButton!
    private var autoStopCheckbox: NSButton!
    private var structureMode: NSSegmentedControl!
    private var librarySearch: NSSearchField!
    private var exportButtons: [NSButton] = []

    private var queue: [QueueItem] = []
    private var library: [LibraryEntry] = []
    private var currentTask: Process?
    private var startupTask: Process?
    private var currentID: UUID?
    private var currentOutputBuffer = ""
    private var currentEventFinished = false
    private var isPausing = false
    private var isQueueRunning = false
    private var progressBatchIDs: Set<UUID> = []
    private var serverRunning = false
    private var healthCheckInFlight = false
    private var serverRecoveryAttempts = 0
    private var healthTimer: Timer?
    private var idleStopTimer: Timer?
    private var totalMemoryGB = 16
    private var activeProfile = PerformanceProfile(key: "balanced", title: "Средняя", serverSequences: 5, sdkWorkers: 5, pageQueue: 7, regionQueue: 144, reservedGB: 6)
    private var loadedResult: [String: Any] = [:]
    private var imagePairs: [[String: Any]] = []
    private var cleanupSheet: NSPanel?
    private var outputFolderLabel: NSTextField!

    private var outputDirectory: String {
        if let saved = UserDefaults.standard.string(forKey: outputDirKey),
           FileManager.default.fileExists(atPath: saved) {
            return saved
        }
        return resultsDir
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        totalMemoryGB = detectedMemoryGB()
        activeProfile = profile(named: loadSavedProfileName())
        loadQueue()
        buildWindow()
        refreshQueueSummary()
        refreshLibrary()
        updateServerStatus()
        healthTimer = Timer.scheduledTimer(withTimeInterval: 3, repeats: true) { [weak self] _ in self?.updateServerStatus() }
    }

    func applicationDidBecomeActive(_ notification: Notification) { window?.makeKeyAndOrderFront(nil) }
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }

    private func buildWindow() {
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1080, height: 790),
                          styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView],
                          backing: .buffered, defer: false)
        window.title = "GLM OCR Local"
        window.titleVisibility = .hidden
        window.titlebarAppearsTransparent = true
        window.titlebarSeparatorStyle = .none
        window.toolbarStyle = .unified
        window.isMovableByWindowBackground = true
        window.isOpaque = false
        window.backgroundColor = .clear
        window.minSize = NSSize(width: 900, height: 680)
        window.center()
        window.delegate = self

        let backdrop = makeSystemGlass(radius: 0) ?? fallbackGlass()
        backdrop.translatesAutoresizingMaskIntoConstraints = true
        window.contentView = backdrop
        let host = glassContent(of: backdrop)

        let root = NSStackView()
        root.orientation = .vertical
        root.alignment = .leading
        root.spacing = 12
        root.translatesAutoresizingMaskIntoConstraints = false
        host.addSubview(root)
        NSLayoutConstraint.activate([
            root.leadingAnchor.constraint(equalTo: host.leadingAnchor, constant: 24),
            root.trailingAnchor.constraint(equalTo: host.trailingAnchor, constant: -24),
            root.topAnchor.constraint(equalTo: host.topAnchor, constant: 50),
            root.bottomAnchor.constraint(equalTo: host.bottomAnchor, constant: -18)
        ])

        root.addArrangedSubview(makeHeader())
        root.addArrangedSubview(makeServerCard())
        tabs = NSTabView()
        tabs.tabViewType = .topTabsBezelBorder
        tabs.controlSize = .regular
        tabs.translatesAutoresizingMaskIntoConstraints = false
        tabs.addTabViewItem(tabItem("Очередь", view: makeQueueTab()))
        tabs.addTabViewItem(tabItem("Предпросмотр", view: makePreviewTab()))
        tabs.addTabViewItem(tabItem("Структура", view: makeStructureTab()))
        tabs.addTabViewItem(tabItem("Изображения", view: makeImagesTab()))
        tabs.addTabViewItem(tabItem("Библиотека", view: makeLibraryTab()))
        root.addArrangedSubview(tabs)
        tabs.widthAnchor.constraint(equalTo: root.widthAnchor).isActive = true
        tabs.heightAnchor.constraint(greaterThanOrEqualToConstant: 510).isActive = true

        buildMainMenu()
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    private func makeHeader() -> NSView {
        let row = NSStackView()
        row.orientation = .horizontal
        row.alignment = .centerY
        row.spacing = 13
        let icon = NSImageView(image: NSImage(named: NSImage.applicationIconName) ?? NSImage())
        icon.imageScaling = .scaleProportionallyUpOrDown
        icon.widthAnchor.constraint(equalToConstant: 42).isActive = true
        icon.heightAnchor.constraint(equalToConstant: 42).isActive = true
        let labels = NSStackView()
        labels.orientation = .vertical
        labels.alignment = .leading
        labels.spacing = 1
        let title = NSTextField(labelWithString: "GLM OCR Local")
        title.font = .systemFont(ofSize: 23, weight: .bold)
        let subtitle = NSTextField(labelWithString: "Документы, таблицы и формулы · полностью локально через MLX/Metal")
        subtitle.font = .systemFont(ofSize: 12.5)
        subtitle.textColor = .secondaryLabelColor
        labels.addArrangedSubview(title)
        labels.addArrangedSubview(subtitle)
        row.addArrangedSubview(icon)
        row.addArrangedSubview(labels)
        row.addArrangedSubview(NSView())
        return row
    }

    private func makeServerCard() -> NSView {
        let view = card()
        let stack = cardStack(in: view)
        stack.orientation = .horizontal
        stack.alignment = .centerY
        stack.spacing = 10
        serverDot = NSImageView(image: symbol("circle.fill", color: .systemOrange))
        serverDot.widthAnchor.constraint(equalToConstant: 11).isActive = true
        serverLabel = NSTextField(labelWithString: "Проверяю сервер…")
        serverLabel.font = .systemFont(ofSize: 12.5, weight: .medium)
        profilePopup = NSPopUpButton(frame: .zero, pullsDown: false)
        profilePopup.addItems(withTitles: ["Авто", "Низкая", "Средняя", "Высокая", "Максимальная", "Экстрим"])
        profilePopup.target = self
        profilePopup.action = #selector(profileChanged)
        selectProfilePopup(activeProfile.key)
        profileDescription = NSTextField(labelWithString: "")
        profileDescription.font = .systemFont(ofSize: 11)
        profileDescription.textColor = .secondaryLabelColor
        updateProfileDescription()
        startButton = button("Запустить", symbol: "play.fill", #selector(startServer))
        stopButton = button("Стоп", symbol: "stop.fill", #selector(stopServer))
        stack.addArrangedSubview(serverDot)
        stack.addArrangedSubview(serverLabel)
        stack.addArrangedSubview(NSView())
        stack.addArrangedSubview(NSTextField(labelWithString: "Нагрузка:"))
        stack.addArrangedSubview(profilePopup)
        stack.addArrangedSubview(profileDescription)
        stack.addArrangedSubview(startButton)
        stack.addArrangedSubview(stopButton)
        return view
    }

    private func makeQueueTab() -> NSView {
        let root = NSStackView()
        root.orientation = .vertical
        root.alignment = .leading
        root.spacing = 10
        root.edgeInsets = NSEdgeInsets(top: 12, left: 12, bottom: 12, right: 12)

        let tools = NSStackView()
        tools.orientation = .horizontal
        tools.spacing = 7
        tools.addArrangedSubview(button("Добавить…", symbol: "plus", #selector(chooseInput), prominent: true))
        upButton = iconButton("arrow.up", "Выше", #selector(moveUp))
        downButton = iconButton("arrow.down", "Ниже", #selector(moveDown))
        removeButton = iconButton("minus", "Убрать из очереди — файлы останутся", #selector(removeSelected))
        retryButton = button("Повторить", symbol: "arrow.clockwise", #selector(retrySelected))
        cleanupButton = button("Очистить…", symbol: "trash", #selector(showCleanup))
        tools.addArrangedSubview(upButton)
        tools.addArrangedSubview(downButton)
        tools.addArrangedSubview(removeButton)
        tools.addArrangedSubview(retryButton)
        tools.addArrangedSubview(cleanupButton)
        tools.addArrangedSubview(NSView())
        tools.addArrangedSubview(button("Показать результаты", symbol: "folder", #selector(openResults)))
        root.addArrangedSubview(tools)
        tools.widthAnchor.constraint(equalTo: root.widthAnchor).isActive = true

        let outputRow = NSStackView()
        outputRow.orientation = .horizontal
        outputRow.alignment = .centerY
        outputRow.spacing = 8
        let outputTitle = NSTextField(labelWithString: "Папка:")
        outputTitle.font = .systemFont(ofSize: 12.5, weight: .semibold)
        outputFolderLabel = NSTextField(labelWithString: outputDirectory)
        outputFolderLabel.font = .systemFont(ofSize: 12)
        outputFolderLabel.lineBreakMode = .byTruncatingMiddle
        outputFolderLabel.textColor = .secondaryLabelColor
        outputFolderLabel.toolTip = outputDirectory
        outputFolderLabel.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
        outputFolderLabel.setContentHuggingPriority(.defaultLow, for: .horizontal)
        outputRow.addArrangedSubview(outputTitle)
        outputRow.addArrangedSubview(outputFolderLabel)
        outputRow.addArrangedSubview(button("Выбрать…", symbol: "folder.badge.plus", #selector(chooseOutputFolder)))
        outputRow.addArrangedSubview(button("Сброс", symbol: "arrow.uturn.backward", #selector(resetOutputFolder)))
        root.addArrangedSubview(outputRow)
        outputRow.widthAnchor.constraint(equalTo: root.widthAnchor).isActive = true

        queueTable = NSTableView()
        queueTable.delegate = self
        queueTable.dataSource = self
        queueTable.rowHeight = 31
        queueTable.headerView = NSTableHeaderView()
        queueTable.usesAlternatingRowBackgroundColors = true
        queueTable.allowsMultipleSelection = true
        queueTable.style = .inset
        queueTable.backgroundColor = .clear
        addColumn(queueTable, id: "name", title: "Документ", width: 430)
        addColumn(queueTable, id: "pages", title: "Страницы", width: 90)
        addColumn(queueTable, id: "status", title: "Состояние", width: 170)
        addColumn(queueTable, id: "progress", title: "Прогресс", width: 150)
        let queueScroll = scrollView(queueTable)
        root.addArrangedSubview(queueScroll)
        queueScroll.widthAnchor.constraint(equalTo: root.widthAnchor).isActive = true
        queueScroll.heightAnchor.constraint(greaterThanOrEqualToConstant: 155).isActive = true

        let settings = NSStackView()
        settings.orientation = .vertical
        settings.alignment = .leading
        settings.spacing = 7
        root.addArrangedSubview(settings)
        settings.widthAnchor.constraint(equalTo: root.widthAnchor).isActive = true
        let enhancementRow = NSStackView()
        enhancementRow.orientation = .horizontal
        enhancementRow.alignment = .centerY
        enhancementRow.spacing = 10
        enhanceCheckbox = NSButton(checkboxWithTitle: "Улучшать картинки после OCR", target: nil, action: nil)
        enhanceCheckbox.state = .on
        enhancementModePopup = NSPopUpButton(frame: .zero, pullsDown: false)
        enhancementModePopup.addItems(withTitles: ["Шум + резкость + MLX x2", "Только шум + резкость"])
        autoStopCheckbox = NSButton(checkboxWithTitle: "Остановить сервер после 10 мин простоя", target: nil, action: nil)
        autoStopCheckbox.state = .on
        enhancementRow.addArrangedSubview(enhanceCheckbox)
        enhancementRow.addArrangedSubview(enhancementModePopup)
        enhancementRow.addArrangedSubview(NSView())
        enhancementRow.addArrangedSubview(autoStopCheckbox)
        settings.addArrangedSubview(enhancementRow)
        enhancementRow.widthAnchor.constraint(equalTo: settings.widthAnchor).isActive = true

        let formatsRow = NSStackView()
        formatsRow.orientation = .horizontal
        formatsRow.alignment = .centerY
        formatsRow.spacing = 12
        let formatsTitle = NSTextField(labelWithString: "Создать:")
        formatsTitle.font = .systemFont(ofSize: 12.5, weight: .semibold)
        fullMDCheckbox = NSButton(checkboxWithTitle: "MD (полный)", target: nil, action: nil)
        splitMDCheckbox = NSButton(checkboxWithTitle: "MD (по главам)", target: nil, action: nil)
        searchablePDFCheckbox = NSButton(checkboxWithTitle: "PDF с поиском", target: nil, action: nil)
        searchablePDFCheckbox.toolTip = "Исходные страницы без пережатия и невидимый текстовый слой по координатам блоков."
        for checkbox in [fullMDCheckbox, splitMDCheckbox, searchablePDFCheckbox] {
            checkbox?.state = .on
        }
        formatsRow.addArrangedSubview(formatsTitle)
        formatsRow.addArrangedSubview(fullMDCheckbox)
        formatsRow.addArrangedSubview(splitMDCheckbox)
        formatsRow.addArrangedSubview(searchablePDFCheckbox)
        formatsRow.addArrangedSubview(NSView())
        settings.addArrangedSubview(formatsRow)
        formatsRow.widthAnchor.constraint(equalTo: settings.widthAnchor).isActive = true

        let pdfRow = NSStackView()
        pdfRow.orientation = .horizontal
        pdfRow.alignment = .centerY
        pdfRow.spacing = 12
        let pdfTitle = NSTextField(labelWithString: "PDF:")
        pdfTitle.font = .systemFont(ofSize: 12.5, weight: .semibold)
        pdfTitle.widthAnchor.constraint(equalToConstant: 55).isActive = true
        pdfRow.addArrangedSubview(pdfTitle)
        pdfRow.addArrangedSubview(NSTextField(labelWithString: "Исходный скан и невидимый текст"))
        pdfRow.addArrangedSubview(NSView())
        settings.addArrangedSubview(pdfRow)
        pdfRow.widthAnchor.constraint(equalTo: settings.widthAnchor).isActive = true

        let actions = NSStackView()
        actions.orientation = .horizontal
        actions.spacing = 8
        runButton = button("Запустить очередь", symbol: "text.viewfinder", #selector(runQueue), prominent: true)
        runButton.keyEquivalent = "\r"
        pauseButton = button("Пауза", symbol: "pause.fill", #selector(pauseQueue))
        pauseButton.isEnabled = false
        actions.addArrangedSubview(runButton)
        actions.addArrangedSubview(pauseButton)
        actions.addArrangedSubview(NSView())
        root.addArrangedSubview(actions)
        actions.widthAnchor.constraint(equalTo: root.widthAnchor).isActive = true

        let progressCard = card()
        let progressStack = cardStack(in: progressCard)
        let progressTop = NSStackView()
        progressTop.orientation = .horizontal
        progressTop.alignment = .centerY
        progressLabel = NSTextField(labelWithString: "Готово к работе")
        progressLabel.font = .systemFont(ofSize: 13.5, weight: .semibold)
        memoryLabel = NSTextField(labelWithString: "")
        memoryLabel.font = .monospacedSystemFont(ofSize: 11, weight: .regular)
        memoryLabel.textColor = .secondaryLabelColor
        progressTop.addArrangedSubview(progressLabel)
        progressTop.addArrangedSubview(NSView())
        progressTop.addArrangedSubview(memoryLabel)
        progressStack.addArrangedSubview(progressTop)
        progressTop.widthAnchor.constraint(equalTo: progressStack.widthAnchor).isActive = true
        progressBar = NSProgressIndicator()
        progressBar.style = .bar
        progressBar.minValue = 0
        progressBar.maxValue = 1
        progressBar.doubleValue = 0
        progressStack.addArrangedSubview(progressBar)
        progressBar.widthAnchor.constraint(equalTo: progressStack.widthAnchor).isActive = true
        detailLabel = NSTextField(wrappingLabelWithString: "Добавьте один или несколько PDF, изображений или папку.")
        detailLabel.font = .systemFont(ofSize: 11.5)
        detailLabel.textColor = .secondaryLabelColor
        progressStack.addArrangedSubview(detailLabel)
        detailLabel.widthAnchor.constraint(equalTo: progressStack.widthAnchor).isActive = true
        logView = NSTextView()
        logView.isEditable = false
        logView.drawsBackground = false
        logView.font = .monospacedSystemFont(ofSize: 10.5, weight: .regular)
        logView.textColor = .secondaryLabelColor
        logView.string = "Здесь отображаются основные события. Технический шум скрыт.\n"
        let logScroll = scrollView(logView)
        logScroll.drawsBackground = false
        progressStack.addArrangedSubview(logScroll)
        logScroll.widthAnchor.constraint(equalTo: progressStack.widthAnchor).isActive = true
        logScroll.heightAnchor.constraint(greaterThanOrEqualToConstant: 80).isActive = true
        root.addArrangedSubview(progressCard)
        progressCard.widthAnchor.constraint(equalTo: root.widthAnchor).isActive = true
        return root
    }

    private func makePreviewTab() -> NSView {
        let root = NSStackView()
        root.orientation = .vertical
        root.alignment = .leading
        root.spacing = 9
        root.edgeInsets = NSEdgeInsets(top: 12, left: 12, bottom: 12, right: 12)
        let bar = NSStackView()
        bar.orientation = .horizontal
        bar.spacing = 7
        let title = NSTextField(labelWithString: "Результат")
        title.font = .systemFont(ofSize: 13, weight: .semibold)
        bar.addArrangedSubview(title)
        bar.addArrangedSubview(NSView())
        for (name, icon, selector) in [
            ("MD", "doc.plaintext", #selector(openMarkdown)),
            ("Главы", "list.bullet", #selector(openChapters)),
            ("PDF", "doc.text", #selector(openSearchablePDF))
        ] {
            let b = button(name, symbol: icon, selector)
            b.isEnabled = false
            exportButtons.append(b)
            bar.addArrangedSubview(b)
        }
        bar.addArrangedSubview(button("Папка", symbol: "folder", #selector(openLoadedResult)))
        root.addArrangedSubview(bar)
        bar.widthAnchor.constraint(equalTo: root.widthAnchor).isActive = true
        let config = WKWebViewConfiguration()
        config.websiteDataStore = .nonPersistent()
        previewWeb = WKWebView(frame: .zero, configuration: config)
        previewWeb.setValue(false, forKey: "drawsBackground")
        previewWeb.loadHTMLString(emptyPreviewHTML("Выберите готовый документ в очереди или библиотеке."), baseURL: nil)
        root.addArrangedSubview(previewWeb)
        previewWeb.widthAnchor.constraint(equalTo: root.widthAnchor).isActive = true
        previewWeb.heightAnchor.constraint(greaterThanOrEqualToConstant: 430).isActive = true
        return root
    }

    private func makeStructureTab() -> NSView {
        let root = NSStackView()
        root.orientation = .vertical
        root.alignment = .leading
        root.spacing = 9
        root.edgeInsets = NSEdgeInsets(top: 12, left: 12, bottom: 12, right: 12)
        structureMode = NSSegmentedControl(labels: ["Все блоки", "Формулы", "Таблицы", "JSON"], trackingMode: .selectOne, target: self, action: #selector(structureModeChanged))
        structureMode.selectedSegment = 0
        root.addArrangedSubview(structureMode)
        structureView = NSTextView()
        structureView.isEditable = false
        structureView.isRichText = false
        structureView.drawsBackground = false
        structureView.font = .monospacedSystemFont(ofSize: 12, weight: .regular)
        structureView.textContainerInset = NSSize(width: 12, height: 12)
        structureView.string = "После OCR здесь появятся типы блоков, формулы, таблицы и исходный JSON."
        let scroll = scrollView(structureView)
        root.addArrangedSubview(scroll)
        scroll.widthAnchor.constraint(equalTo: root.widthAnchor).isActive = true
        scroll.heightAnchor.constraint(greaterThanOrEqualToConstant: 430).isActive = true
        return root
    }

    private func makeImagesTab() -> NSView {
        let root = NSStackView()
        root.orientation = .vertical
        root.alignment = .leading
        root.spacing = 9
        root.edgeInsets = NSEdgeInsets(top: 12, left: 12, bottom: 12, right: 12)
        let tools = NSStackView()
        tools.orientation = .horizontal
        tools.spacing = 8
        imagePopup = NSPopUpButton(frame: .zero, pullsDown: false)
        imagePopup.addItem(withTitle: "Нет извлечённых изображений")
        imagePopup.target = self
        imagePopup.action = #selector(imageChanged)
        compareSlider = NSSlider(value: 1, minValue: 0, maxValue: 1, target: self, action: #selector(compareChanged))
        compareSlider.widthAnchor.constraint(equalToConstant: 150).isActive = true
        tools.addArrangedSubview(imagePopup)
        tools.addArrangedSubview(NSView())
        tools.addArrangedSubview(NSTextField(labelWithString: "Оригинал"))
        tools.addArrangedSubview(compareSlider)
        tools.addArrangedSubview(NSTextField(labelWithString: "Улучшено"))
        root.addArrangedSubview(tools)
        tools.widthAnchor.constraint(equalTo: root.widthAnchor).isActive = true
        let pair = NSStackView()
        pair.orientation = .horizontal
        pair.distribution = .fillEqually
        pair.spacing = 12
        let left = imagePanel(title: "До", image: &originalImage, caption: &originalCaption)
        let right = imagePanel(title: "После", image: &enhancedImage, caption: &enhancedCaption)
        pair.addArrangedSubview(left)
        pair.addArrangedSubview(right)
        root.addArrangedSubview(pair)
        pair.widthAnchor.constraint(equalTo: root.widthAnchor).isActive = true
        pair.heightAnchor.constraint(greaterThanOrEqualToConstant: 420).isActive = true
        return root
    }

    private func makeLibraryTab() -> NSView {
        let root = NSStackView()
        root.orientation = .vertical
        root.alignment = .leading
        root.spacing = 9
        root.edgeInsets = NSEdgeInsets(top: 12, left: 12, bottom: 12, right: 12)
        let tools = NSStackView()
        tools.orientation = .horizontal
        tools.spacing = 8
        librarySearch = NSSearchField()
        librarySearch.placeholderString = "Поиск по распознанному тексту"
        librarySearch.sendsSearchStringImmediately = true
        librarySearch.target = self
        librarySearch.action = #selector(searchLibrary)
        librarySearch.widthAnchor.constraint(greaterThanOrEqualToConstant: 330).isActive = true
        tools.addArrangedSubview(librarySearch)
        tools.addArrangedSubview(button("Открыть", symbol: "eye", #selector(openLibrarySelection), prominent: true))
        tools.addArrangedSubview(button("Папка", symbol: "folder", #selector(openLibraryFolder)))
        tools.addArrangedSubview(NSView())
        tools.addArrangedSubview(button("Обновить", symbol: "arrow.clockwise", #selector(refreshLibraryAction)))
        tools.addArrangedSubview(button("Очистить…", symbol: "trash", #selector(showCleanup)))
        root.addArrangedSubview(tools)
        tools.widthAnchor.constraint(equalTo: root.widthAnchor).isActive = true
        libraryTable = NSTableView()
        libraryTable.delegate = self
        libraryTable.dataSource = self
        libraryTable.rowHeight = 34
        libraryTable.doubleAction = #selector(openLibrarySelection)
        libraryTable.target = self
        libraryTable.usesAlternatingRowBackgroundColors = true
        libraryTable.style = .inset
        libraryTable.backgroundColor = .clear
        addColumn(libraryTable, id: "libTitle", title: "Документ", width: 370)
        addColumn(libraryTable, id: "libPages", title: "Страницы", width: 75)
        addColumn(libraryTable, id: "libSource", title: "Исходный файл / совпадение", width: 480)
        let scroll = scrollView(libraryTable)
        root.addArrangedSubview(scroll)
        scroll.widthAnchor.constraint(equalTo: root.widthAnchor).isActive = true
        scroll.heightAnchor.constraint(greaterThanOrEqualToConstant: 430).isActive = true
        return root
    }

    private func imagePanel(title: String, image: inout NSImageView!, caption: inout NSTextField!) -> NSView {
        let panel = card()
        let stack = cardStack(in: panel)
        let label = NSTextField(labelWithString: title)
        label.font = .systemFont(ofSize: 13, weight: .semibold)
        image = NSImageView()
        image.imageScaling = .scaleProportionallyUpOrDown
        image.wantsLayer = true
        image.layer?.backgroundColor = NSColor.black.withAlphaComponent(0.04).cgColor
        image.layer?.cornerRadius = 16
        caption = NSTextField(wrappingLabelWithString: "Нет изображения")
        caption.font = .systemFont(ofSize: 11)
        caption.textColor = .secondaryLabelColor
        caption.lineBreakMode = .byTruncatingMiddle
        stack.addArrangedSubview(label)
        stack.addArrangedSubview(image)
        stack.addArrangedSubview(caption)
        image.widthAnchor.constraint(equalTo: stack.widthAnchor).isActive = true
        image.heightAnchor.constraint(greaterThanOrEqualToConstant: 340).isActive = true
        caption.widthAnchor.constraint(equalTo: stack.widthAnchor).isActive = true
        return panel
    }

    private func buildMainMenu() {
        let main = NSMenu()
        let appItem = NSMenuItem()
        let appMenu = NSMenu()
        appMenu.addItem(withTitle: "О GLM OCR Local", action: #selector(NSApplication.orderFrontStandardAboutPanel(_:)), keyEquivalent: "")
        appMenu.addItem(.separator())
        appMenu.addItem(withTitle: "Завершить GLM OCR Local", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        appItem.submenu = appMenu
        main.addItem(appItem)
        let fileItem = NSMenuItem()
        let fileMenu = NSMenu(title: "Файл")
        let add = fileMenu.addItem(withTitle: "Добавить документы…", action: #selector(chooseInput), keyEquivalent: "o")
        add.target = self
        let output = fileMenu.addItem(withTitle: "Папка результатов…", action: #selector(chooseOutputFolder), keyEquivalent: "d")
        output.target = self
        fileMenu.addItem(.separator())
        let cleanup = fileMenu.addItem(withTitle: "Очистить…", action: #selector(showCleanup), keyEquivalent: "k")
        cleanup.target = self
        let run = fileMenu.addItem(withTitle: "Запустить очередь", action: #selector(runQueue), keyEquivalent: "r")
        run.target = self
        let pause = fileMenu.addItem(withTitle: "Пауза", action: #selector(pauseQueue), keyEquivalent: ".")
        pause.target = self
        fileItem.submenu = fileMenu
        main.addItem(fileItem)
        NSApp.mainMenu = main
    }

    @objc private func chooseInput() {
        let panel = NSOpenPanel()
        panel.title = "Добавьте PDF, изображения или папки"
        panel.prompt = "Добавить в очередь"
        panel.canChooseFiles = true
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = true
        if panel.runModal() == .OK {
            var sources: [URL] = []
            for url in panel.urls {
                if (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true {
                    if let enumerator = FileManager.default.enumerator(at: url, includingPropertiesForKeys: [.isRegularFileKey], options: [.skipsHiddenFiles]) {
                        for case let file as URL in enumerator where supported(file) { sources.append(file) }
                    }
                } else if supported(url) { sources.append(url) }
            }
            let existing = Set(queue.filter { $0.status != "complete" }.map(\.source))
            for source in sources where !existing.contains(source.path) {
                let stamp = Self.timestamp.string(from: Date())
                let unique = UUID().uuidString.prefix(6)
                let stem = safeFilename(source.deletingPathExtension().lastPathComponent)
                let output = URL(fileURLWithPath: outputDirectory).appendingPathComponent("\(stem)-\(stamp)-\(unique)").path
                let pages = source.pathExtension.lowercased() == "pdf" ? (PDFDocument(url: source)?.pageCount ?? 0) : 1
                queue.append(QueueItem(id: UUID(), source: source.path, output: output, status: "pending", progress: 0,
                                       pagesDone: 0, pagesTotal: pages, etaSeconds: nil, message: "Ожидает", addedAt: Date()))
            }
            saveQueue()
            queueTable.reloadData()
            refreshQueueSummary()
        }
    }

    @objc private func runQueue() {
        idleStopTimer?.invalidate()
        if let selected = selectedQueueIndex(), queue[selected].status == "paused" || queue[selected].status == "error" {
            queue[selected].status = "pending"
        }
        guard queue.contains(where: { $0.status == "pending" || $0.status == "paused" }) else {
            alert("В очереди нет новых или приостановленных документов.")
            return
        }
        for index in queue.indices where queue[index].status == "paused" { queue[index].status = "pending" }
        progressBatchIDs = Set(queue.filter { $0.status == "pending" }.map(\.id))
        isQueueRunning = true
        isPausing = false
        runButton.isEnabled = false
        pauseButton.isEnabled = true
        pauseButton.title = "Пауза"
        pauseButton.image = symbol("pause.fill", color: .labelColor)
        lockSettings(true)
        saveQueue()
        if serverRunning {
            processNext()
        } else {
            progressLabel.stringValue = "Запускаю локальную модель…"
            detailLabel.stringValue = "Первый запуск сервера обычно занимает несколько секунд."
            runSimpleProcess(startScript) { [weak self] code, _ in
                guard let self else { return }
                if code == 0 { self.serverRunning = true; self.processNext() }
                else { self.finishQueue(message: "Не удалось запустить локальный сервер") }
            }
        }
    }

    private func processNext() {
        guard isQueueRunning else { return }
        guard let index = queue.firstIndex(where: { $0.status == "pending" }) else {
            finishQueue(message: "Очередь завершена")
            return
        }
        var item = queue[index]
        guard FileManager.default.fileExists(atPath: item.source) else {
            queue[index].status = "error"
            queue[index].message = "Исходный файл не найден"
            saveQueue(); queueTable.reloadData(); processNext(); return
        }
        try? FileManager.default.createDirectory(atPath: item.output, withIntermediateDirectories: true)
        item.status = "running"
        item.message = item.progress > 0 ? "Продолжение" : "Подготовка"
        queue[index] = item
        currentID = item.id
        currentOutputBuffer = ""
        currentEventFinished = false
        queueTable.reloadData()
        refreshQueueSummary()
        saveQueue()
        append("\n▶ \(item.name)\n")

        let task = Process()
        let pipe = Pipe()
        task.executableURL = URL(fileURLWithPath: sdkPython)
        let formats = selectedExportFormats()
        task.arguments = [jobRunner, "--input", item.source, "--output", item.output,
                          "--enhance", enhancementMode(), formats.isEmpty ? "--no-exports" : "--exports",
                          "--export-formats", formats.joined(separator: ",")]
        task.currentDirectoryURL = URL(fileURLWithPath: installDir)
        var env = ProcessInfo.processInfo.environment
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        task.environment = env
        task.standardOutput = pipe
        task.standardError = pipe
        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
            DispatchQueue.main.async { self?.consumeManagerOutput(text) }
        }
        task.terminationHandler = { [weak self] process in
            pipe.fileHandleForReading.readabilityHandler = nil
            DispatchQueue.main.async {
                guard let self else { return }
                self.currentTask = nil
                let id = self.currentID
                self.currentID = nil
                if !self.currentEventFinished, let id, let row = self.queue.firstIndex(where: { $0.id == id }) {
                    if self.isPausing || process.terminationStatus == 130 {
                        self.queue[row].status = "paused"
                        self.queue[row].message = "Можно продолжить"
                    } else if process.terminationStatus != 0 {
                        self.queue[row].status = "error"
                        self.queue[row].message = "Ошибка — можно повторить"
                    }
                }
                if process.terminationStatus == 75, let id, let row = self.queue.firstIndex(where: { $0.id == id }) {
                    self.queue[row].status = "paused"
                    self.queue[row].message = "Сервер перезапускается — продолжу автоматически"
                    self.saveQueue(); self.queueTable.reloadData(); self.refreshQueueSummary()
                    self.recoverServerAndResume(row: row)
                    return
                }
                if self.currentEventFinished, let id, let row = self.queue.firstIndex(where: { $0.id == id }), self.queue[row].status == "error" {
                    self.isQueueRunning = false
                    self.runButton.isEnabled = true
                    self.pauseButton.isEnabled = false
                    self.lockSettings(false)
                    self.progressLabel.stringValue = "Очередь остановлена из-за ошибки"
                    self.detailLabel.stringValue = "Исправьте причину и нажмите «Повторить» или «Запустить очередь»."
                    self.saveQueue(); self.queueTable.reloadData(); self.refreshQueueSummary()
                    return
                }
                self.saveQueue(); self.queueTable.reloadData(); self.refreshQueueSummary()
                if self.isPausing { self.finishQueue(message: "Очередь приостановлена", paused: true) }
                else { self.processNext() }
            }
        }
        do { try task.run(); currentTask = task }
        catch {
            queue[index].status = "error"; queue[index].message = error.localizedDescription
            currentID = nil; saveQueue(); processNext()
        }
    }

    private func consumeManagerOutput(_ chunk: String) {
        currentOutputBuffer += chunk.replacingOccurrences(of: "\r", with: "\n")
        let lines = currentOutputBuffer.components(separatedBy: "\n")
        currentOutputBuffer = lines.last ?? ""
        for raw in lines.dropLast() {
            guard raw.hasPrefix("GLMOCR_EVENT "),
                  let data = String(raw.dropFirst(13)).data(using: .utf8),
                  let event = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let type = event["type"] as? String else { continue }
            handleEvent(type, event)
        }
    }

    private func handleEvent(_ type: String, _ event: [String: Any]) {
        guard let id = currentID, let row = queue.firstIndex(where: { $0.id == id }) else { return }
        if let total = event["pages_total"] as? Int { queue[row].pagesTotal = total }
        if let done = event["pages_done"] as? Int { queue[row].pagesDone = done }
        if let eta = event["eta_seconds"] as? Int { queue[row].etaSeconds = eta }
        let enhancementEnabled = enhanceCheckbox.state == .on
        let exportEnabled = !selectedExportFormats().isEmpty
        let enhancementShare = enhancementEnabled ? 0.10 : 0.0
        let exportShare = exportEnabled ? 0.08 : 0.0
        let optimizeShare = 0.06
        let ocrEnd = 1.0 - enhancementShare - exportShare - optimizeShare
        let enhancementEnd = ocrEnd + enhancementShare
        let exportEnd = enhancementEnd + exportShare
        func ocrProgress() {
            if let fraction = event["fraction"] as? Double {
                queue[row].progress = min(ocrEnd, max(0, fraction) * ocrEnd)
            }
        }
        switch type {
        case "job_started":
            ocrProgress()
            queue[row].message = "Распознавание"
        case "resumed":
            queue[row].message = "Продолжение с сохранённого места"
            append("↻ Продолжаю с последнего готового блока.\n")
        case "chunk_started":
            ocrProgress()
            let start = event["page_start"] as? Int ?? 1
            let end = event["page_end"] as? Int ?? start
            queue[row].message = "Страницы \(start)–\(end)"
        case "heartbeat":
            let stage = event["stage"] as? String ?? "ocr"
            if stage == "ocr" {
                ocrProgress()
                queue[row].message = "Распознавание"
            }
            if let free = event["memory_free_percent"] as? Int {
                memoryLabel.stringValue = "свободно памяти: \(free)%"
                memoryLabel.textColor = free < 20 ? .systemOrange : .secondaryLabelColor
            }
        case "memory_wait":
            queue[row].message = "Пауза для памяти"
            detailLabel.stringValue = event["message"] as? String ?? "Жду освобождения памяти"
            append("◷ Жду освобождения памяти перед следующим блоком.\n")
        case "chunk_completed":
            ocrProgress()
            queue[row].message = "Блок сохранён"
            append("✓ Сохранено: \(queue[row].pagesDone) из \(queue[row].pagesTotal) стр.\n")
        case "stage":
            let stage = event["stage"] as? String ?? ""
            if stage == "assembled" {
                queue[row].etaSeconds = nil
                queue[row].progress = max(queue[row].progress, ocrEnd)
                queue[row].message = "Markdown и JSON готовы"
            } else if stage == "enhancement" {
                queue[row].etaSeconds = nil
                queue[row].progress = max(queue[row].progress, ocrEnd)
                queue[row].message = "Улучшение изображений"
            } else if stage == "export" {
                queue[row].etaSeconds = nil
                queue[row].progress = max(queue[row].progress, enhancementEnd)
                queue[row].message = "Экспорт результатов"
            } else if stage == "optimize" {
                queue[row].etaSeconds = nil
                queue[row].progress = max(queue[row].progress, exportEnd)
                queue[row].message = "Оптимизация изображений"
            }
        case "enhancement_total":
            let total = event["total"] as? Int ?? 0
            queue[row].progress = max(queue[row].progress, ocrEnd)
            queue[row].message = total > 0 ? "Улучшение рисунков: 0 из \(total)" : "Рисунки для улучшения не найдены"
        case "enhancement_progress":
            let done = event["done"] as? Int ?? 0
            let total = max(1, event["total"] as? Int ?? 1)
            queue[row].progress = ocrEnd + enhancementShare * min(1, Double(done) / Double(total))
            queue[row].message = "Улучшение рисунков: \(done) из \(total)"
        case "export_started":
            queue[row].progress = max(queue[row].progress, enhancementEnd)
            queue[row].message = "Подготовка выбранных форматов"
        case "export_format_started":
            let format = event["format"] as? String ?? ""
            let done = event["done"] as? Int ?? 0
            let total = max(1, event["total"] as? Int ?? 1)
            queue[row].progress = enhancementEnd + exportShare * Double(done) / Double(total)
            queue[row].message = "Создаю: \(formatTitle(format))"
        case "optimize_started":
            let total = event["total"] as? Int ?? 0
            queue[row].progress = max(queue[row].progress, exportEnd)
            queue[row].message = total > 0 ? "Оптимизация картинок: 0 из \(total)" : "Картинки для упаковки не найдены"
        case "optimize_progress":
            let done = event["done"] as? Int ?? 0
            let total = max(1, event["total"] as? Int ?? 1)
            queue[row].progress = exportEnd + optimizeShare * min(1, Double(done) / Double(total))
            queue[row].message = "Оптимизация картинок: \(done) из \(total)"
        case "optimize_done":
            let saved = event["saved_bytes"] as? Int ?? 0
            queue[row].progress = max(queue[row].progress, exportEnd + optimizeShare)
            let megabytes = Double(saved) / 1_048_576
            queue[row].message = saved > 0 ? String(format: "Картинки легче на %.1f МБ", megabytes) : "Картинки уже компактные"
            append(saved > 0 ? String(format: "✓ Изображения упакованы без потери пикселей, −%.1f МБ.\n", megabytes) : "✓ Упаковка изображений ничего не изменила.\n")
        case "clean_pdf_progress":
            let pagesDone = event["pages_done"] as? Int ?? 0
            let pagesTotal = max(1, event["pages_total"] as? Int ?? 1)
            let formatDone = event["format_done"] as? Int ?? 0
            let formatTotal = max(1, event["format_total"] as? Int ?? 1)
            let withinFormat = min(1, Double(pagesDone) / Double(pagesTotal))
            queue[row].progress = enhancementEnd + exportShare * (Double(formatDone) + withinFormat) / Double(formatTotal)
            queue[row].message = "Чистый PDF: страница \(pagesDone) из \(pagesTotal)"
        case "searchable_pdf_progress":
            let pagesDone = event["pages_done"] as? Int ?? 0
            let pagesTotal = max(1, event["pages_total"] as? Int ?? 1)
            let formatDone = event["format_done"] as? Int ?? 0
            let formatTotal = max(1, event["format_total"] as? Int ?? 1)
            let withinFormat = min(1, Double(pagesDone) / Double(pagesTotal))
            queue[row].progress = enhancementEnd + exportShare * (Double(formatDone) + withinFormat) / Double(formatTotal)
            queue[row].message = "PDF с текстом: страница \(pagesDone) из \(pagesTotal)"
        case "export_progress":
            let format = event["format"] as? String ?? ""
            let done = event["done"] as? Int ?? 0
            let total = max(1, event["total"] as? Int ?? 1)
            queue[row].progress = enhancementEnd + exportShare * Double(done) / Double(total)
            queue[row].message = "Готово: \(formatTitle(format))"
        case "export_done":
            queue[row].progress = max(queue[row].progress, 0.99)
            queue[row].message = "Все выбранные форматы готовы"
        case "export_error":
            let message = event["message"] as? String ?? "Не удалось создать один из форматов"
            queue[row].message = "Экспорт: предупреждение"
            append("! \(message)\n")
        case "complete":
            queue[row].status = "complete"
            queue[row].progress = 1
            queue[row].pagesDone = queue[row].pagesTotal
            queue[row].message = "Готово"
            currentEventFinished = true
            append("✓ Готово: \(queue[row].name)\n")
            loadResult(queue[row].output, switchTab: true)
            refreshLibrary()
        case "job_paused":
            queue[row].status = "paused"
            queue[row].message = "Можно продолжить"
            currentEventFinished = true
        case "server_lost":
            queue[row].status = "paused"
            queue[row].message = "Сервер недоступен — перезапускаю"
            currentEventFinished = true
            append("↻ Локальный сервер остановился. Перезапускаю и продолжу этот блок.\n")
        case "error":
            queue[row].status = "error"
            queue[row].message = event["message"] as? String ?? "Ошибка"
            currentEventFinished = true
            append("✕ \(queue[row].message)\n")
        case "log":
            if let message = event["message"] as? String { append("! \(message)\n") }
        default: break
        }
        saveQueue(); queueTable.reloadData(); refreshQueueSummary()
    }

    @objc private func pauseQueue() {
        guard let task = currentTask else {
            finishQueue(message: "Очередь приостановлена", paused: true); return
        }
        isPausing = true
        pauseButton.title = "Останавливаю…"
        pauseButton.isEnabled = false
        progressLabel.stringValue = "Сохраняю и приостанавливаю…"
        detailLabel.stringValue = "Текущий незавершённый блок будет безопасно остановлен; готовые блоки сохранятся."
        task.interrupt()
    }

    private func finishQueue(message: String, paused: Bool = false) {
        isQueueRunning = false
        isPausing = false
        currentTask = nil
        currentID = nil
        runButton.isEnabled = queue.contains(where: { $0.status == "pending" || $0.status == "paused" || $0.status == "error" })
        pauseButton.isEnabled = false
        pauseButton.title = "Пауза"
        pauseButton.image = symbol("pause.fill", color: .labelColor)
        lockSettings(false)
        progressLabel.stringValue = message
        refreshQueueSummary()
        saveQueue()
        if !paused { NSSound(named: .init("Glass"))?.play(); scheduleIdleStop() }
    }

    private func recoverServerAndResume(row: Int) {
        guard isQueueRunning, queue.indices.contains(row) else { return }
        guard serverRecoveryAttempts < 3 else {
            queue[row].status = "error"
            queue[row].message = "Сервер не удалось восстановить — можно повторить"
            serverRecoveryAttempts = 0
            finishQueue(message: "Не удалось автоматически восстановить сервер")
            return
        }
        serverRecoveryAttempts += 1
        progressLabel.stringValue = "Перезапускаю локальную модель…"
        detailLabel.stringValue = "Попытка \(serverRecoveryAttempts) из 3 · готовые блоки сохранены"
        runSimpleProcess(startScript) { [weak self] code, output in
            guard let self else { return }
            self.append(output)
            if code == 0 {
                self.serverRunning = true
                self.serverRecoveryAttempts = 0
                self.queue[row].status = "pending"
                self.queue[row].message = "Продолжение после перезапуска"
                self.saveQueue(); self.queueTable.reloadData()
                self.processNext()
            } else {
                DispatchQueue.main.asyncAfter(deadline: .now() + 3) { self.recoverServerAndResume(row: row) }
            }
        }
    }

    private func refreshQueueSummary() {
        guard progressBar != nil else { return }
        let tracked = progressBatchIDs.isEmpty ? queue : queue.filter { progressBatchIDs.contains($0.id) }
        let count = max(1, tracked.count)
        let overall = tracked.reduce(0.0) { $0 + $1.progress } / Double(count)
        progressBar.doubleValue = overall
        let complete = queue.filter { $0.status == "complete" }.count
        let errors = queue.filter { $0.status == "error" }.count
        if let running = queue.first(where: { $0.status == "running" }) {
            progressLabel.stringValue = running.name
            let pageText = running.pagesTotal > 0 ? "Страницы: \(running.pagesDone) из \(running.pagesTotal)" : "Подготовка"
            let eta = running.etaSeconds.map { " · осталось примерно \(duration($0))" } ?? ""
            detailLabel.stringValue = "\(pageText) · \(Int(running.progress * 100))%\(eta) · \(running.message)"
        } else if !isQueueRunning {
            detailLabel.stringValue = "В очереди: \(queue.count) · готово: \(complete) · ошибок: \(errors). Приостановленные задания можно продолжить."
        }
        runButton?.isEnabled = !isQueueRunning && queue.contains { $0.status == "pending" || $0.status == "paused" || $0.status == "error" }
        updateQueueButtons()
    }

    @objc private func moveUp() { moveSelected(-1) }
    @objc private func moveDown() { moveSelected(1) }
    private func moveSelected(_ delta: Int) {
        guard !isQueueRunning, let row = selectedQueueIndex() else { return }
        let target = row + delta
        guard target >= 0, target < queue.count else { return }
        queue.swapAt(row, target); saveQueue(); queueTable.reloadData(); queueTable.selectRowIndexes(IndexSet(integer: target), byExtendingSelection: false)
    }

    @objc private func removeSelected() {
        guard !isQueueRunning else { return }
        let rows = queueTable.selectedRowIndexes.sorted(by: >)
        for row in rows where row < queue.count { queue.remove(at: row) }
        saveQueue(); queueTable.reloadData(); refreshQueueSummary()
    }

    @objc private func retrySelected() {
        for row in queueTable.selectedRowIndexes where row < queue.count && ["error", "paused"].contains(queue[row].status) {
            queue[row].status = "pending"; queue[row].message = queue[row].progress > 0 ? "Продолжит с сохранённого места" : "Ожидает"
        }
        saveQueue(); queueTable.reloadData(); refreshQueueSummary()
    }

    @objc private func showCleanup() {
        guard !isQueueRunning else { alert("Сначала поставьте очередь на паузу."); return }
        let selectedRows = queueTable.selectedRowIndexes.filter { queue.indices.contains($0) }
        var selectedPaths = selectedRows.compactMap { queue[$0].status == "complete" && isCompletedResult(queue[$0].output) ? queue[$0].output : nil }
        if let selectedTab = tabs.selectedTabViewItem, tabs.indexOfTabViewItem(selectedTab) == 4,
           library.indices.contains(libraryTable.selectedRow), isCompletedResult(library[libraryTable.selectedRow].path) {
            selectedPaths.append(library[libraryTable.selectedRow].path)
        }
        selectedPaths = uniquePaths(selectedPaths)
        let allPaths = uniquePaths(
            queue.compactMap { $0.status == "complete" && isCompletedResult($0.output) ? $0.output : nil } +
            library.compactMap { isCompletedResult($0.path) ? $0.path : nil }
        )
        cleanupButton?.isEnabled = false
        progressLabel?.stringValue = "Считаю объём для очистки…"
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self else { return }
            let selectedBytes = selectedPaths.reduce(Int64(0)) { $0 + self.directoryBytes(URL(fileURLWithPath: $1)) }
            let allBytes = allPaths.reduce(Int64(0)) { $0 + self.directoryBytes(URL(fileURLWithPath: $1)) }
            let logBytes = self.directoryBytes(URL(fileURLWithPath: logFile))
            DispatchQueue.main.async {
                self.cleanupButton?.isEnabled = true
                if self.queueTable != nil { self.refreshQueueSummary() }
                self.presentCleanupChooser(selectedPaths: selectedPaths, allPaths: allPaths,
                                           selectedBytes: selectedBytes, allBytes: allBytes, logBytes: logBytes)
            }
        }
    }

    private func presentCleanupChooser(selectedPaths: [String], allPaths: [String], selectedBytes: Int64, allBytes: Int64, logBytes: Int64) {
        guard cleanupSheet == nil else { return }
        let panel = NSPanel(contentRect: NSRect(x: 0, y: 0, width: 680, height: 410),
                            styleMask: [.titled], backing: .buffered, defer: false)
        panel.titlebarAppearsTransparent = true
        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.title = "Очистка"
        panel.isReleasedWhenClosed = false
        panel.animationBehavior = .documentWindow
        panel.minSize = NSSize(width: 680, height: 410)
        panel.maxSize = NSSize(width: 680, height: 410)

        let background = makeSystemGlass(radius: 0) ?? fallbackGlass()
        background.translatesAutoresizingMaskIntoConstraints = true
        panel.contentView = background
        let host = glassContent(of: background)

        let root = NSStackView()
        root.orientation = .vertical
        root.alignment = .leading
        root.spacing = 16
        root.translatesAutoresizingMaskIntoConstraints = false
        host.addSubview(root)
        NSLayoutConstraint.activate([
            root.leadingAnchor.constraint(equalTo: host.leadingAnchor, constant: 28),
            root.trailingAnchor.constraint(equalTo: host.trailingAnchor, constant: -28),
            root.topAnchor.constraint(equalTo: host.topAnchor, constant: 24),
            root.bottomAnchor.constraint(equalTo: host.bottomAnchor, constant: -22)
        ])

        let header = NSStackView()
        header.orientation = .horizontal; header.alignment = .centerY; header.spacing = 14
        let headerIcon = NSImageView(image: symbol("externaldrive.badge.minus", color: .controlAccentColor))
        headerIcon.symbolConfiguration = NSImage.SymbolConfiguration(pointSize: 28, weight: .medium)
        headerIcon.widthAnchor.constraint(equalToConstant: 42).isActive = true
        headerIcon.heightAnchor.constraint(equalToConstant: 42).isActive = true
        let headerText = NSStackView(); headerText.orientation = .vertical; headerText.alignment = .leading; headerText.spacing = 3
        let title = NSTextField(labelWithString: "Безопасная очистка")
        title.font = .systemFont(ofSize: 20, weight: .bold)
        let subtitle = NSTextField(wrappingLabelWithString: "Освободите место, не затрагивая исходные документы и локальную модель.")
        subtitle.font = .systemFont(ofSize: 12.5); subtitle.textColor = .secondaryLabelColor
        headerText.addArrangedSubview(title); headerText.addArrangedSubview(subtitle)
        header.addArrangedSubview(headerIcon); header.addArrangedSubview(headerText)
        root.addArrangedSubview(header); header.widthAnchor.constraint(equalTo: root.widthAnchor).isActive = true

        let choiceStack = NSStackView(); choiceStack.orientation = .vertical; choiceStack.alignment = .leading; choiceStack.spacing = 6
        let choiceLabel = NSTextField(labelWithString: "Что очистить")
        choiceLabel.font = .systemFont(ofSize: 12, weight: .semibold)
        let chooser = NSPopUpButton(frame: .zero, pullsDown: false)
        chooser.addItems(withTitles: [
            "Удалить служебные файлы выбранных — результат сохранить",
            "Переместить выбранные результаты в Корзину",
            "Переместить все завершённые результаты в Корзину",
            "Очистить технический журнал сервера",
            "Только убрать завершённые записи из очереди"
        ])
        chooser.item(at: 0)?.isEnabled = !selectedPaths.isEmpty
        chooser.item(at: 1)?.isEnabled = !selectedPaths.isEmpty
        chooser.item(at: 2)?.isEnabled = !allPaths.isEmpty
        chooser.item(at: 4)?.isEnabled = queue.contains(where: { $0.status == "complete" })
        if selectedPaths.isEmpty { chooser.selectItem(at: allPaths.isEmpty ? 3 : 2) }
        choiceStack.addArrangedSubview(choiceLabel); choiceStack.addArrangedSubview(chooser)
        root.addArrangedSubview(choiceStack)
        choiceStack.widthAnchor.constraint(equalTo: root.widthAnchor).isActive = true
        chooser.widthAnchor.constraint(equalTo: choiceStack.widthAnchor).isActive = true

        let stats = NSStackView(); stats.orientation = .horizontal; stats.distribution = .fillEqually; stats.spacing = 10
        stats.addArrangedSubview(cleanupStat(title: "ВЫБРАНО", value: "\(selectedPaths.count) · \(formatBytes(selectedBytes))"))
        stats.addArrangedSubview(cleanupStat(title: "ВСЕ ГОТОВЫЕ", value: "\(allPaths.count) · \(formatBytes(allBytes))"))
        stats.addArrangedSubview(cleanupStat(title: "ЖУРНАЛ", value: formatBytes(logBytes)))
        root.addArrangedSubview(stats); stats.widthAnchor.constraint(equalTo: root.widthAnchor).isActive = true

        let details = NSTextField(wrappingLabelWithString: "Служебные файлы — это блоки возобновления и визуализации структуры. Markdown, JSON, изображения и экспорты сохранятся. Полные результаты перемещаются в Корзину и могут быть восстановлены.")
        details.font = .systemFont(ofSize: 12); details.textColor = .secondaryLabelColor
        details.maximumNumberOfLines = 0; details.preferredMaxLayoutWidth = 624
        root.addArrangedSubview(details); details.widthAnchor.constraint(equalTo: root.widthAnchor).isActive = true

        let safety = NSStackView(); safety.orientation = .horizontal; safety.alignment = .centerY; safety.spacing = 7
        let lock = NSImageView(image: symbol("lock.shield.fill", color: .systemGreen))
        lock.widthAnchor.constraint(equalToConstant: 16).isActive = true
        let safetyText = NSTextField(labelWithString: "Исходные PDF и изображения, модель, приложение и окружения не удаляются.")
        safetyText.font = .systemFont(ofSize: 11.5, weight: .medium); safetyText.textColor = .secondaryLabelColor
        safety.addArrangedSubview(lock); safety.addArrangedSubview(safetyText)
        root.addArrangedSubview(safety)

        let actions = NSStackView(); actions.orientation = .horizontal; actions.alignment = .centerY; actions.spacing = 10
        actions.addArrangedSubview(NSView())
        let cancel = NSButton(title: "Отмена", target: self, action: #selector(cancelCleanupSheet))
        cancel.bezelStyle = .rounded; cancel.keyEquivalent = "\u{1b}"; cancel.widthAnchor.constraint(equalToConstant: 108).isActive = true
        let proceed = NSButton(title: "Продолжить", target: self, action: #selector(acceptCleanupSheet))
        proceed.bezelStyle = .rounded; proceed.bezelColor = .controlAccentColor; proceed.contentTintColor = .white
        proceed.font = .systemFont(ofSize: 13, weight: .semibold); proceed.keyEquivalent = "\r"
        proceed.widthAnchor.constraint(equalToConstant: 128).isActive = true
        actions.addArrangedSubview(cancel); actions.addArrangedSubview(proceed)
        root.addArrangedSubview(actions); actions.widthAnchor.constraint(equalTo: root.widthAnchor).isActive = true

        cleanupSheet = panel
        window.beginSheet(panel) { [weak self, weak chooser] response in
            guard let self else { return }
            self.cleanupSheet = nil
            guard response == .OK, let chooser else { return }
            DispatchQueue.main.async {
                switch chooser.indexOfSelectedItem {
                case 0: self.trimSupportFiles(paths: selectedPaths)
                case 1: self.trashResults(paths: selectedPaths, knownBytes: selectedBytes)
                case 2: self.trashResults(paths: allPaths, knownBytes: allBytes)
                case 3: self.clearServerLog(knownBytes: logBytes)
                case 4: self.clearCompletedQueueEntries()
                default: break
                }
            }
        }
    }

    private func cleanupStat(title: String, value: String) -> NSView {
        let box = NSBox(); box.boxType = .custom; box.cornerRadius = 8
        box.fillColor = .controlBackgroundColor; box.borderColor = .separatorColor.withAlphaComponent(0.45)
        box.borderWidth = 0.5; box.contentViewMargins = NSSize(width: 11, height: 8)
        let stack = NSStackView(); stack.orientation = .vertical; stack.alignment = .leading; stack.spacing = 2
        let heading = NSTextField(labelWithString: title); heading.font = .systemFont(ofSize: 9.5, weight: .semibold); heading.textColor = .tertiaryLabelColor
        let amount = NSTextField(labelWithString: value); amount.font = .monospacedDigitSystemFont(ofSize: 12.5, weight: .medium)
        stack.addArrangedSubview(heading); stack.addArrangedSubview(amount)
        box.contentView = stack; box.heightAnchor.constraint(equalToConstant: 52).isActive = true
        return box
    }

    @objc private func acceptCleanupSheet() {
        guard let sheet = cleanupSheet else { return }
        window.endSheet(sheet, returnCode: .OK)
    }

    @objc private func cancelCleanupSheet() {
        guard let sheet = cleanupSheet else { return }
        window.endSheet(sheet, returnCode: .cancel)
    }

    private func trimSupportFiles(paths: [String]) {
        guard !paths.isEmpty else { return }
        inspectSupport(paths: paths) { [weak self] bytes in
            guard let self else { return }
            let confirmation = NSAlert()
            confirmation.messageText = "Удалить служебные файлы?"
            confirmation.informativeText = "Будет освобождено примерно \(self.formatBytes(bytes)). Итоговые Markdown, JSON, изображения и экспорты сохранятся. После этого повторно продолжать уже завершённое задание по блокам не потребуется и будет невозможно."
            confirmation.alertStyle = .warning
            confirmation.addButton(withTitle: "Удалить служебные файлы")
            confirmation.addButton(withTitle: "Отмена")
            guard confirmation.runModal() == .alertFirstButtonReturn else { return }
            self.runCleanupSequential(paths: paths, index: 0, reclaimed: 0)
        }
    }

    private func inspectSupport(paths: [String], completion: @escaping (Int64) -> Void) {
        DispatchQueue.global(qos: .userInitiated).async {
            var bytes: Int64 = 0
            for path in paths {
                let task = Process(); let pipe = Pipe()
                task.executableURL = URL(fileURLWithPath: sdkPython)
                task.arguments = [cleanupScript, "inspect", path]
                task.standardOutput = pipe; task.standardError = FileHandle.nullDevice
                do {
                    try task.run(); task.waitUntilExit()
                    let data = pipe.fileHandleForReading.readDataToEndOfFile()
                    if let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any], let number = object["support_bytes"] as? NSNumber { bytes += number.int64Value }
                } catch { }
            }
            DispatchQueue.main.async { completion(bytes) }
        }
    }

    private func runCleanupSequential(paths: [String], index: Int, reclaimed: Int64) {
        guard index < paths.count else {
            append("✓ Служебные файлы удалены. Освобождено примерно \(formatBytes(reclaimed)).\n")
            alert("Очистка завершена. Освобождено примерно \(formatBytes(reclaimed)). Итоговые результаты сохранены.")
            return
        }
        let path = paths[index]
        runCapture(sdkPython, [cleanupScript, "trim", path]) { [weak self] code, output in
            guard let self else { return }
            var freed: Int64 = 0
            if let data = output.data(using: .utf8), let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any], let number = object["reclaimed_bytes"] as? NSNumber { freed = number.int64Value }
            if code != 0 { self.append("! Не удалось очистить \(URL(fileURLWithPath: path).lastPathComponent).\n") }
            self.runCleanupSequential(paths: paths, index: index + 1, reclaimed: reclaimed + freed)
        }
    }

    private func trashResults(paths: [String], knownBytes: Int64) {
        let safePaths = uniquePaths(paths.filter { isCompletedResult($0) })
        guard !safePaths.isEmpty else { return }
        let bytes = knownBytes
        let confirmation = NSAlert()
        confirmation.messageText = safePaths.count == 1 ? "Переместить результат в Корзину?" : "Переместить результаты в Корзину?"
        confirmation.informativeText = "Будет освобождено после очистки Корзины примерно \(formatBytes(bytes)). Исходные документы останутся на месте. Результаты можно восстановить из Корзины."
        confirmation.alertStyle = .warning
        confirmation.addButton(withTitle: "Переместить в Корзину")
        confirmation.addButton(withTitle: "Отмена")
        guard confirmation.runModal() == .alertFirstButtonReturn else { return }

        var trashedPaths = Set<String>()
        var failures: [String] = []
        for path in safePaths {
            guard let url = resultURL(path), isCompletedResult(path) else { failures.append(URL(fileURLWithPath: path).lastPathComponent); continue }
            do {
                try FileManager.default.trashItem(at: url, resultingItemURL: nil)
                trashedPaths.insert(url.path)
                removeFromLibrarySynchronously(path)
            } catch { failures.append("\(URL(fileURLWithPath: path).lastPathComponent): \(error.localizedDescription)") }
        }
        queue.removeAll { trashedPaths.contains(URL(fileURLWithPath: $0.output).standardizedFileURL.resolvingSymlinksInPath().path) }
        if let root = loadedResult["root"] as? String, !FileManager.default.fileExists(atPath: root) { resetLoadedResult() }
        saveQueue(); queueTable.reloadData(); refreshQueueSummary(); refreshLibrary()
        append("✓ В Корзину перемещено: \(trashedPaths.count). Освободится после очистки Корзины: до \(formatBytes(bytes)).\n")
        if failures.isEmpty { alert("Готово: \(trashedPaths.count) результат(ов) перемещено в Корзину. Исходные документы сохранены.") }
        else { alert("Часть результатов не удалось переместить:\n\(failures.joined(separator: "\n"))") }
    }

    private func clearServerLog(knownBytes: Int64) {
        let bytes = knownBytes
        let confirmation = NSAlert()
        confirmation.messageText = "Очистить журнал сервера?"
        confirmation.informativeText = "Будет освобождено примерно \(formatBytes(bytes)). Результаты OCR и модель не затрагиваются."
        confirmation.alertStyle = .warning
        confirmation.addButton(withTitle: "Очистить журнал")
        confirmation.addButton(withTitle: "Отмена")
        guard confirmation.runModal() == .alertFirstButtonReturn else { return }
        runCapture(sdkPython, [cleanupScript, "clear-logs"]) { [weak self] code, _ in
            guard let self else { return }
            if code == 0 { self.append("✓ Журнал сервера очищен.\n"); self.alert("Журнал очищен. Результаты OCR сохранены.") }
            else { self.alert("Не удалось очистить журнал.") }
        }
    }

    private func clearCompletedQueueEntries() {
        let count = queue.filter { $0.status == "complete" }.count
        guard count > 0 else { return }
        let confirmation = NSAlert()
        confirmation.messageText = "Убрать завершённые задания из очереди?"
        confirmation.informativeText = "Будут убраны только \(count) записей очереди. Все результаты на диске и записи библиотеки сохранятся."
        confirmation.addButton(withTitle: "Убрать из очереди")
        confirmation.addButton(withTitle: "Отмена")
        guard confirmation.runModal() == .alertFirstButtonReturn else { return }
        queue.removeAll { $0.status == "complete" }
        saveQueue(); queueTable.reloadData(); refreshQueueSummary()
        append("✓ Завершённые записи убраны из очереди; файлы сохранены.\n")
    }

    private func resultURL(_ path: String) -> URL? {
        let root = URL(fileURLWithPath: resultsDir).standardizedFileURL.resolvingSymlinksInPath()
        let candidate = URL(fileURLWithPath: path).standardizedFileURL.resolvingSymlinksInPath()
        guard candidate != root, candidate.deletingLastPathComponent() == root,
              FileManager.default.fileExists(atPath: candidate.path) else { return nil }
        if let values = try? URL(fileURLWithPath: path).resourceValues(forKeys: [.isSymbolicLinkKey]), values.isSymbolicLink == true { return nil }
        return candidate
    }

    private func isCompletedResult(_ path: String) -> Bool {
        guard resultURL(path) != nil,
              let data = try? Data(contentsOf: URL(fileURLWithPath: path).appendingPathComponent("job-state.json")),
              let state = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return false }
        return state["status"] as? String == "complete"
    }

    private func uniquePaths(_ paths: [String]) -> [String] {
        var seen = Set<String>()
        return paths.filter {
            guard let url = resultURL($0) else { return false }
            return seen.insert(url.path).inserted
        }.map { resultURL($0)!.path }
    }

    private func directoryBytes(_ url: URL) -> Int64 {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory) else { return 0 }
        if !isDirectory.boolValue {
            let values = try? url.resourceValues(forKeys: [.totalFileAllocatedSizeKey, .fileAllocatedSizeKey, .fileSizeKey])
            return Int64(values?.totalFileAllocatedSize ?? values?.fileAllocatedSize ?? values?.fileSize ?? 0)
        }
        guard let enumerator = FileManager.default.enumerator(at: url, includingPropertiesForKeys: [.isRegularFileKey, .totalFileAllocatedSizeKey, .fileAllocatedSizeKey, .fileSizeKey], options: [.skipsHiddenFiles]) else { return 0 }
        var total: Int64 = 0
        for case let file as URL in enumerator {
            let values = try? file.resourceValues(forKeys: [.isRegularFileKey, .totalFileAllocatedSizeKey, .fileAllocatedSizeKey, .fileSizeKey])
            if values?.isRegularFile == true { total += Int64(values?.totalFileAllocatedSize ?? values?.fileAllocatedSize ?? values?.fileSize ?? 0) }
        }
        return total
    }

    private func formatBytes(_ bytes: Int64) -> String { ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file) }
    private func removeFromLibrarySynchronously(_ path: String) {
        let task = Process()
        task.executableURL = URL(fileURLWithPath: sdkPython)
        task.arguments = [libraryScript, "remove", path]
        task.currentDirectoryURL = URL(fileURLWithPath: installDir)
        task.standardOutput = FileHandle.nullDevice
        task.standardError = FileHandle.nullDevice
        do { try task.run(); task.waitUntilExit() } catch { }
    }
    private func resetLoadedResult() {
        loadedResult = [:]; imagePairs = []
        previewWeb.loadHTMLString(emptyPreviewHTML("Результат перемещён в Корзину."), baseURL: nil)
        structureView.string = "Результат перемещён в Корзину."
        imagePopup.removeAllItems(); imagePopup.addItem(withTitle: "Нет извлечённых изображений")
        showImagePair(0); exportButtons.forEach { $0.isEnabled = false }
    }

    @objc private func startServer() {
        idleStopTimer?.invalidate()
        append("Запускаю локальную модель…\n")
        runSimpleProcess(startScript) { [weak self] code, output in
            self?.append(output)
            if code != 0 { self?.alert("Не удалось запустить сервер. Подробности есть в журнале.") }
            self?.updateServerStatus()
        }
    }

    @objc private func stopServer() {
        guard !isQueueRunning else { alert("Сначала поставьте очередь на паузу."); return }
        runSimpleProcess(stopScript) { [weak self] _, output in self?.append(output); self?.updateServerStatus() }
    }

    @objc private func profileChanged() {
        guard !isQueueRunning else { selectProfilePopup(activeProfile.key); return }
        let keys = ["auto", "low", "balanced", "high", "max", "extreme"]
        let next = profile(named: keys[max(0, profilePopup.indexOfSelectedItem)])
        let restart = serverRunning && next.serverSequences != activeProfile.serverSequences
        activeProfile = next
        saveProfile(next)
        updateProfileDescription()
        append("Профиль нагрузки: \(next.title).\n")
        if restart {
            profilePopup.isEnabled = false
            runSimpleProcess(stopScript) { [weak self] stopCode, output in
                guard let self else { return }
                self.append(output)
                guard stopCode == 0 else {
                    self.profilePopup.isEnabled = true
                    self.alert("Сервер ещё завершает текущий запрос. Профиль сохранён и будет применён при следующем запуске.")
                    self.updateServerStatus()
                    return
                }
                self.runSimpleProcess(startScript) { _, startOutput in
                    self.append(startOutput); self.profilePopup.isEnabled = true; self.updateServerStatus()
                }
            }
        }
    }

    private func updateServerStatus() {
        guard !healthCheckInFlight else { return }
        healthCheckInFlight = true
        DispatchQueue.global(qos: .utility).async { [weak self] in
            guard let self else { return }
            let task = Process()
            task.executableURL = URL(fileURLWithPath: "/usr/bin/curl")
            task.arguments = ["--silent", "--fail", "--connect-timeout", "1", "--max-time", "2", "http://127.0.0.1:8080/health"]
            task.standardOutput = FileHandle.nullDevice
            task.standardError = FileHandle.nullDevice
            try? task.run(); task.waitUntilExit()
            let healthy = task.terminationStatus == 0
            let pidURL = URL(fileURLWithPath: installDir + "/glm-ocr-server.pid")
            let pid = (try? String(contentsOf: pidURL, encoding: .utf8))?.trimmingCharacters(in: .whitespacesAndNewlines)
            var processAlive = false
            if let pid, let value = Int32(pid), value > 1, kill(value, 0) == 0 {
                let commandTask = Process(); let commandPipe = Pipe()
                commandTask.executableURL = URL(fileURLWithPath: "/bin/ps")
                commandTask.arguments = ["-p", pid, "-o", "command="]
                commandTask.standardOutput = commandPipe; commandTask.standardError = FileHandle.nullDevice
                try? commandTask.run(); commandTask.waitUntilExit()
                let command = String(data: commandPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
                processAlive = command.contains(installDir + "/.venv-mlx/bin/mlx_vlm.server")
            }
            DispatchQueue.main.async {
                self.healthCheckInFlight = false
                self.serverRunning = healthy || processAlive
                if healthy {
                    self.serverLabel.stringValue = "MLX/Metal · localhost:8080 · документы никуда не отправляются"
                    self.serverDot.image = self.symbol("circle.fill", color: .systemGreen)
                } else if processAlive {
                    self.serverLabel.stringValue = "MLX/Metal занят обработкой или загружается · localhost:8080"
                    self.serverDot.image = self.symbol("circle.fill", color: .systemYellow)
                } else {
                    self.serverLabel.stringValue = "Локальная модель остановлена"
                    self.serverDot.image = self.symbol("circle.fill", color: .systemOrange)
                }
                self.startButton.isEnabled = !self.serverRunning
                self.stopButton.isEnabled = self.serverRunning && !self.isQueueRunning
            }
        }
    }

    private func scheduleIdleStop() {
        idleStopTimer?.invalidate()
        guard autoStopCheckbox.state == .on else { return }
        idleStopTimer = Timer.scheduledTimer(withTimeInterval: 600, repeats: false) { [weak self] _ in
            guard let self, !self.isQueueRunning, self.serverRunning else { return }
            self.runSimpleProcess(stopScript) { _, _ in self.updateServerStatus() }
        }
    }

    private func loadResult(_ root: String, switchTab: Bool) {
        runCapture(sdkPython, [resultInfoScript, root]) { [weak self] code, output in
            guard let self, code == 0, let data = output.data(using: .utf8),
                  let info = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return }
            self.loadedResult = info
            self.updatePreview()
            self.updateStructure()
            self.updateImages()
            if switchTab { self.tabs.selectTabViewItem(at: 1) }
        }
    }

    private func updatePreview() {
        let html = loadedResult["html"] as? String ?? ""
        let markdown = loadedResult["markdown"] as? String ?? ""
        if !html.isEmpty, FileManager.default.fileExists(atPath: html) {
            let url = URL(fileURLWithPath: html)
            previewWeb.loadFileURL(url, allowingReadAccessTo: URL(fileURLWithPath: loadedResult["root"] as? String ?? installDir))
        } else if let text = try? String(contentsOfFile: markdown, encoding: .utf8) {
            previewWeb.loadHTMLString(markdownPreviewHTML(text), baseURL: URL(fileURLWithPath: (loadedResult["final_dir"] as? String ?? installDir)))
        } else {
            previewWeb.loadHTMLString(emptyPreviewHTML("Предпросмотр ещё не создан."), baseURL: nil)
        }
        let paths = loadedResult["exports"] as? [String] ?? []
        let markers = [".md", "_chapters", ".pdf"]
        for (index, marker) in markers.enumerated() where exportButtons.indices.contains(index) {
            exportButtons[index].isEnabled = paths.contains { $0.lowercased().contains(marker) || URL(fileURLWithPath: $0).lastPathComponent.lowercased().hasSuffix(marker) }
        }
    }

    @objc private func structureModeChanged() { updateStructure() }
    private func updateStructure() {
        guard !loadedResult.isEmpty else { return }
        let labels = loadedResult["labels"] as? [String: Any] ?? [:]
        var value: Any
        switch structureMode.selectedSegment {
        case 1: value = loadedResult["formulas"] ?? []
        case 2: value = loadedResult["tables"] ?? []
        case 3:
            let path = loadedResult["json"] as? String ?? ""
            structureView.string = (try? String(contentsOfFile: path, encoding: .utf8)) ?? "JSON не найден"
            return
        default:
            value = ["Типы блоков": labels, "Блоки": loadedResult["blocks"] ?? []]
        }
        if let data = try? JSONSerialization.data(withJSONObject: value, options: [.prettyPrinted, .sortedKeys]), let text = String(data: data, encoding: .utf8) {
            structureView.string = text
        }
    }

    private func updateImages() {
        imagePairs = loadedResult["images"] as? [[String: Any]] ?? []
        imagePopup.removeAllItems()
        if imagePairs.isEmpty { imagePopup.addItem(withTitle: "Нет извлечённых изображений") }
        else { for (index, pair) in imagePairs.enumerated() { imagePopup.addItem(withTitle: "\(index + 1). \(URL(fileURLWithPath: pair["original"] as? String ?? "").lastPathComponent)") } }
        imagePopup.selectItem(at: 0)
        showImagePair(0)
    }

    private func loadPreviewImage(_ path: String) -> NSImage? {
        guard !path.isEmpty, FileManager.default.fileExists(atPath: path) else { return nil }
        let url = URL(fileURLWithPath: path) as CFURL
        if let source = CGImageSourceCreateWithURL(url, nil),
           let image = CGImageSourceCreateImageAtIndex(source, 0, nil) {
            return NSImage(cgImage: image, size: .zero)
        }
        return NSImage(contentsOfFile: path)
    }

    @objc private func imageChanged() { showImagePair(imagePopup.indexOfSelectedItem) }
    private func showImagePair(_ index: Int) {
        guard imagePairs.indices.contains(index) else {
            originalImage.image = nil; enhancedImage.image = nil; originalCaption.stringValue = "Нет изображения"; enhancedCaption.stringValue = "Нет изображения"; return
        }
        let pair = imagePairs[index]
        let original = pair["original"] as? String ?? ""
        let enhanced = pair["enhanced"] as? String ?? ""
        originalImage.image = loadPreviewImage(original)
        enhancedImage.image = loadPreviewImage(enhanced)
        originalCaption.stringValue = original
        enhancedCaption.stringValue = enhanced.isEmpty ? "Для этого изображения улучшенная копия не создавалась" : "\(enhanced)\n\(pair["method"] as? String ?? "")"
        compareChanged()
    }

    @objc private func compareChanged() {
        originalImage.alphaValue = CGFloat(1.0 - compareSlider.doubleValue * 0.35)
        enhancedImage.alphaValue = CGFloat(0.65 + compareSlider.doubleValue * 0.35)
    }

    private func refreshLibrary(query: String = "") {
        let arguments = query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? [libraryScript, "list"] : [libraryScript, "search", query]
        runCapture(sdkPython, arguments) { [weak self] code, output in
            guard let self, code == 0, let data = output.data(using: .utf8), let rows = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] else { return }
            self.library = rows.map { LibraryEntry(title: $0["title"] as? String ?? "Без названия", path: $0["path"] as? String ?? "", source: $0["source"] as? String ?? "", pages: $0["pages"] as? Int ?? 0, snippet: ($0["snippet"] as? String ?? "").replacingOccurrences(of: "<b>", with: "").replacingOccurrences(of: "</b>", with: "")) }
            self.libraryTable?.reloadData()
        }
    }

    @objc private func searchLibrary() { refreshLibrary(query: librarySearch.stringValue) }
    @objc private func refreshLibraryAction() { refreshLibrary(query: librarySearch.stringValue) }
    @objc private func openLibrarySelection() {
        let row = libraryTable.selectedRow
        guard library.indices.contains(row) else { return }
        loadResult(library[row].path, switchTab: true)
    }
    @objc private func openLibraryFolder() {
        let row = libraryTable.selectedRow
        guard library.indices.contains(row) else { return }
        NSWorkspace.shared.open(URL(fileURLWithPath: library[row].path))
    }

    @objc private func chooseOutputFolder() {
        let panel = NSOpenPanel()
        panel.title = "Куда сохранять результаты"
        panel.prompt = "Выбрать"
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.canCreateDirectories = true
        panel.allowsMultipleSelection = false
        panel.directoryURL = URL(fileURLWithPath: outputDirectory, isDirectory: true)
        guard panel.runModal() == .OK, let url = panel.url else { return }
        let previous = outputDirectory
        UserDefaults.standard.set(url.path, forKey: outputDirKey)
        retargetPendingOutputs(from: previous)
        showOutputDirectory()
    }

    @objc private func resetOutputFolder() {
        let previous = outputDirectory
        UserDefaults.standard.removeObject(forKey: outputDirKey)
        retargetPendingOutputs(from: previous)
        showOutputDirectory()
    }

    private func retargetPendingOutputs(from previous: String) {
        let oldRoot = URL(fileURLWithPath: previous).standardizedFileURL.path
        let newRoot = URL(fileURLWithPath: outputDirectory).standardizedFileURL.path
        guard oldRoot != newRoot else { return }
        for index in queue.indices where queue[index].status == "pending" {
            let url = URL(fileURLWithPath: queue[index].output).standardizedFileURL
            if url.deletingLastPathComponent().path == oldRoot {
                queue[index].output = URL(fileURLWithPath: newRoot).appendingPathComponent(url.lastPathComponent).path
            }
        }
        saveQueue()
        queueTable?.reloadData()
    }

    private func showOutputDirectory() {
        outputFolderLabel?.stringValue = outputDirectory
        outputFolderLabel?.toolTip = outputDirectory
    }

    @objc private func openResults() { NSWorkspace.shared.open(URL(fileURLWithPath: outputDirectory)) }
    @objc private func openLoadedResult() {
        let root = loadedResult["root"] as? String ?? resultsDir
        NSWorkspace.shared.open(URL(fileURLWithPath: root))
    }
    @objc private func openMarkdown() { openExport(".md") }
    @objc private func openChapters() {
        guard let paths = loadedResult["exports"] as? [String],
              let path = paths.first(where: { $0.contains("_chapters") }) else { return }
        NSWorkspace.shared.open(URL(fileURLWithPath: path))
    }
    @objc private func openSearchablePDF() { openExport(".pdf") }
    private func openExport(_ suffix: String) {
        guard let paths = loadedResult["exports"] as? [String], let path = paths.first(where: { $0.lowercased().hasSuffix(suffix) }) else { return }
        NSWorkspace.shared.open(URL(fileURLWithPath: path))
    }

    func numberOfRows(in tableView: NSTableView) -> Int { tableView === queueTable ? queue.count : library.count }
    func tableView(_ tableView: NSTableView, viewFor tableColumn: NSTableColumn?, row: Int) -> NSView? {
        let id = tableColumn?.identifier.rawValue ?? ""
        let cell = (tableView.makeView(withIdentifier: NSUserInterfaceItemIdentifier(id), owner: self) as? NSTableCellView) ?? {
            let view = NSTableCellView()
            view.identifier = NSUserInterfaceItemIdentifier(id)
            let text = NSTextField(labelWithString: "")
            text.lineBreakMode = .byTruncatingMiddle
            text.translatesAutoresizingMaskIntoConstraints = false
            view.addSubview(text); view.textField = text
            NSLayoutConstraint.activate([text.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 6), text.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -6), text.centerYAnchor.constraint(equalTo: view.centerYAnchor)])
            return view
        }()
        if tableView === queueTable {
            guard queue.indices.contains(row) else { return cell }
            let item = queue[row]
            switch id {
            case "name": cell.textField?.stringValue = item.name
            case "pages": cell.textField?.stringValue = item.pagesTotal > 0 ? "\(item.pagesDone)/\(item.pagesTotal)" : "—"
            case "status": cell.textField?.stringValue = statusTitle(item.status) + (item.message.isEmpty ? "" : " · \(item.message)")
            case "progress": cell.textField?.stringValue = "\(Int(item.progress * 100))%" + (item.etaSeconds.map { " · ~\(duration($0))" } ?? "")
            default: break
            }
            cell.textField?.textColor = item.status == "error" ? .systemRed : (item.status == "complete" ? .systemGreen : .labelColor)
        } else {
            guard library.indices.contains(row) else { return cell }
            let item = library[row]
            switch id {
            case "libTitle": cell.textField?.stringValue = item.title
            case "libPages": cell.textField?.stringValue = "\(item.pages)"
            default: cell.textField?.stringValue = item.snippet.isEmpty ? item.source : item.snippet
            }
        }
        cell.textField?.font = .systemFont(ofSize: 11.5)
        return cell
    }

    func tableViewSelectionDidChange(_ notification: Notification) {
        if notification.object as? NSTableView === queueTable {
            updateQueueButtons()
            if let row = selectedQueueIndex(), queue[row].status == "complete" { loadResult(queue[row].output, switchTab: false) }
        }
    }

    private func updateQueueButtons() {
        guard queueTable != nil else { return }
        let row = selectedQueueIndex()
        let editable = !isQueueRunning && row != nil
        upButton.isEnabled = editable && row! > 0
        downButton.isEnabled = editable && row! < queue.count - 1
        removeButton.isEnabled = editable
        retryButton.isEnabled = !isQueueRunning && queueTable.selectedRowIndexes.contains { queue.indices.contains($0) && ["error", "paused"].contains(queue[$0].status) }
        cleanupButton.isEnabled = !isQueueRunning
    }

    private func selectedQueueIndex() -> Int? { queueTable.selectedRow >= 0 && queue.indices.contains(queueTable.selectedRow) ? queueTable.selectedRow : nil }
    private func enhancementMode() -> String { enhanceCheckbox.state == .off ? "off" : (enhancementModePopup.indexOfSelectedItem == 0 ? "hybrid" : "safe") }
    private func selectedExportFormats() -> [String] {
        var formats: [String] = []
        if fullMDCheckbox.state == .on { formats.append("md") }
        if splitMDCheckbox.state == .on { formats.append("split_md") }
        if searchablePDFCheckbox.state == .on { formats.append("searchable_pdf") }
        return formats
    }
    private func formatTitle(_ format: String) -> String {
        switch format {
        case "md": return "MD (полный)"
        case "split_md": return "MD (по главам)"
        case "html": return "HTML"
        case "docx": return "DOCX"
        case "searchable_pdf": return "PDF с исходным видом"
        case "clean_pdf": return "чистый цифровой PDF"
        case "zip": return "ZIP-архив"
        default: return format
        }
    }
    private func lockSettings(_ locked: Bool) {
        profilePopup.isEnabled = !locked
        enhanceCheckbox.isEnabled = !locked
        enhancementModePopup.isEnabled = !locked
        for checkbox in [fullMDCheckbox, splitMDCheckbox, searchablePDFCheckbox] {
            checkbox?.isEnabled = !locked
        }
        stopButton.isEnabled = serverRunning && !locked
    }

    private func loadQueue() {
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: queueFile)), var saved = try? JSONDecoder().decode([QueueItem].self, from: data) else { return }
        for index in saved.indices where saved[index].status == "running" { saved[index].status = "paused"; saved[index].message = "Приложение было закрыто — можно продолжить" }
        queue = saved
        saveQueue()
    }
    private func saveQueue() {
        guard let data = try? JSONEncoder().encode(queue) else { return }
        let target = URL(fileURLWithPath: queueFile)
        let temp = target.appendingPathExtension("tmp")
        try? data.write(to: temp, options: .atomic)
        try? FileManager.default.removeItem(at: target)
        try? FileManager.default.moveItem(at: temp, to: target)
    }

    private func runSimpleProcess(_ executable: String, completion: @escaping (Int32, String) -> Void) { runCapture(executable, [], completion: completion) }
    private func runCapture(_ executable: String, _ arguments: [String], completion: @escaping (Int32, String) -> Void) {
        DispatchQueue.global(qos: .userInitiated).async {
            let task = Process(); let pipe = Pipe()
            task.executableURL = URL(fileURLWithPath: executable); task.arguments = arguments
            task.currentDirectoryURL = URL(fileURLWithPath: installDir)
            task.standardOutput = pipe; task.standardError = pipe
            do { try task.run(); task.waitUntilExit(); let data = pipe.fileHandleForReading.readDataToEndOfFile(); let output = String(data: data, encoding: .utf8) ?? ""; DispatchQueue.main.async { completion(task.terminationStatus, output) } }
            catch { DispatchQueue.main.async { completion(1, error.localizedDescription) } }
        }
    }

    private func detectedMemoryGB() -> Int {
        var value: UInt64 = 0; var size = MemoryLayout<UInt64>.size
        return sysctlbyname("hw.memsize", &value, &size, nil, 0) == 0 ? max(8, Int((Double(value) / 1_073_741_824.0).rounded())) : 16
    }

    private func profile(named key: String) -> PerformanceProfile {
        let m = totalMemoryGB
        switch key {
        case "low": return PerformanceProfile(key: key, title: "Низкая", serverSequences: m <= 16 ? 2 : 3, sdkWorkers: m <= 16 ? 2 : 3, pageQueue: 3, regionQueue: 48, reservedGB: m <= 16 ? 8 : max(9, m / 3))
        case "high": return PerformanceProfile(key: key, title: "Высокая", serverSequences: m <= 16 ? 6 : min(12, m / 2), sdkWorkers: m <= 16 ? 6 : min(12, m / 2), pageQueue: m <= 16 ? 9 : 16, regionQueue: m <= 16 ? 192 : 320, reservedGB: m <= 16 ? 5 : max(7, m / 4))
        case "max": return PerformanceProfile(key: key, title: "Максимальная", serverSequences: m <= 16 ? 7 : min(16, m / 2), sdkWorkers: m <= 16 ? 7 : min(16, m / 2), pageQueue: m <= 16 ? 11 : 20, regionQueue: m <= 16 ? 224 : 384, reservedGB: m <= 16 ? 4 : max(6, m / 5))
        case "extreme": return PerformanceProfile(key: key, title: "Экстрим", serverSequences: m <= 8 ? 3 : (m <= 16 ? 9 : min(24, max(12, m * 3 / 4))), sdkWorkers: m <= 8 ? 3 : (m <= 16 ? 9 : min(24, max(12, m * 3 / 4))), pageQueue: m <= 8 ? 6 : (m <= 16 ? 14 : min(32, m)), regionQueue: m <= 8 ? 96 : (m <= 16 ? 320 : min(640, m * 16)), reservedGB: m <= 16 ? 2 : max(3, m / 12))
        case "auto": return PerformanceProfile(key: key, title: "Авто", serverSequences: m <= 16 ? 4 : min(9, m / 3), sdkWorkers: m <= 16 ? 2 : min(9, m / 3), pageQueue: m <= 16 ? 2 : 12, regionQueue: m <= 16 ? 32 : 240, reservedGB: m <= 16 ? 3 : max(8, m / 3))
        default: return PerformanceProfile(key: "balanced", title: "Средняя", serverSequences: m <= 16 ? 5 : min(10, m / 3), sdkWorkers: m <= 16 ? 5 : min(10, m / 3), pageQueue: m <= 16 ? 7 : 14, regionQueue: m <= 16 ? 144 : 288, reservedGB: m <= 16 ? 6 : max(8, m / 4))
        }
    }

    private func loadSavedProfileName() -> String {
        guard let content = try? String(contentsOfFile: profileFile, encoding: .utf8) else { return "balanced" }
        let value = content.split(separator: "\n").first(where: { $0.hasPrefix("PROFILE=") }).map { String($0.dropFirst(8)) } ?? "balanced"
        return value == "fast" ? "high" : value
    }
    private func saveProfile(_ p: PerformanceProfile) {
        let text = "PROFILE=\(p.key)\nSERVER_SEQS=\(p.serverSequences)\nSDK_WORKERS=\(p.sdkWorkers)\nPAGE_QUEUE=\(p.pageQueue)\nREGION_QUEUE=\(p.regionQueue)\nTOTAL_MEMORY_GB=\(totalMemoryGB)\nRESERVED_MEMORY_GB=\(p.reservedGB)\n"
        try? text.write(toFile: profileFile, atomically: true, encoding: .utf8)
    }
    private func selectProfilePopup(_ key: String) { profilePopup.selectItem(at: ["auto": 0, "low": 1, "balanced": 2, "high": 3, "max": 4, "extreme": 5][key] ?? 2) }
    private func updateProfileDescription() {
        if activeProfile.key == "extreme" {
            profileDescription.stringValue = "\(totalMemoryGB) ГБ · почти весь ресурс · запас ≥\(activeProfile.reservedGB) ГБ"
            profileDescription.textColor = .systemOrange
        } else {
            profileDescription.stringValue = "\(totalMemoryGB) ГБ · запас ≥\(activeProfile.reservedGB) ГБ"
            profileDescription.textColor = .secondaryLabelColor
        }
    }

    private func card() -> NSView {
        if let glass = makeSystemGlass(radius: 18) { return glass }
        return fallbackGlass(radius: 18)
    }

    /// На macOS 26 это настоящее Liquid Glass. На 14–15 тот же рисунок из материала окна.
    private func makeSystemGlass(radius: CGFloat) -> NSView? {
        guard let kind = NSClassFromString("NSGlassEffectView") as? NSView.Type else { return nil }
        let view = kind.init(frame: .zero)
        if view.responds(to: NSSelectorFromString("setCornerRadius:")) {
            view.setValue(radius, forKey: "cornerRadius")
        }
        if view.responds(to: NSSelectorFromString("setStyle:")) {
            view.setValue(0, forKey: "style")
        }
        return view
    }

    private func glassContent(of view: NSView) -> NSView {
        guard view.responds(to: NSSelectorFromString("contentView")) else { return view }
        if let content = view.value(forKey: "contentView") as? NSView { return content }
        let inner = NSView(frame: view.bounds)
        view.setValue(inner, forKey: "contentView")
        return (view.value(forKey: "contentView") as? NSView) ?? view
    }

    private func fallbackGlass(radius: CGFloat = 0) -> NSVisualEffectView {
        let view = NSVisualEffectView()
        view.material = .popover
        view.blendingMode = .behindWindow
        view.state = .followsWindowActiveState
        view.wantsLayer = true
        if radius > 0 {
            view.layer?.cornerRadius = radius
            view.layer?.cornerCurve = .continuous
            view.layer?.masksToBounds = true
            view.layer?.borderWidth = 0.8
            view.layer?.borderColor = NSColor.white.withAlphaComponent(0.32).cgColor
        }
        return view
    }

    private func cardStack(in card: NSView) -> NSStackView {
        let host = glassContent(of: card)
        let s = NSStackView(); s.orientation = .vertical; s.alignment = .leading; s.spacing = 8; s.translatesAutoresizingMaskIntoConstraints = false; host.addSubview(s)
        NSLayoutConstraint.activate([s.leadingAnchor.constraint(equalTo: host.leadingAnchor, constant: 14), s.trailingAnchor.constraint(equalTo: host.trailingAnchor, constant: -14), s.topAnchor.constraint(equalTo: host.topAnchor, constant: 12), s.bottomAnchor.constraint(equalTo: host.bottomAnchor, constant: -12)])
        return s
    }
    private func button(_ title: String, symbol name: String? = nil, _ action: Selector, prominent: Bool = false) -> NSButton {
        let b = NSButton(title: title, target: self, action: action); b.bezelStyle = .rounded
        if let name { b.image = symbol(name, color: prominent ? .white : .labelColor); b.imagePosition = .imageLeading }
        if prominent { b.bezelColor = .controlAccentColor; b.contentTintColor = .white; b.font = .systemFont(ofSize: 12.5, weight: .semibold) }
        return b
    }
    private func iconButton(_ name: String, _ help: String, _ action: Selector) -> NSButton { let b = button("", symbol: name, action); b.toolTip = help; b.imagePosition = .imageOnly; return b }
    private func symbol(_ name: String, color: NSColor) -> NSImage { let image = NSImage(systemSymbolName: name, accessibilityDescription: nil) ?? NSImage(); return image.withSymbolConfiguration(.init(paletteColors: [color])) ?? image }
    private func tabItem(_ title: String, view: NSView) -> NSTabViewItem { let item = NSTabViewItem(); item.label = title; item.view = view; return item }
    private func scrollView(_ document: NSView) -> NSScrollView {
        let scroll = NSScrollView()
        scroll.hasVerticalScroller = true
        scroll.hasHorizontalScroller = false
        scroll.autohidesScrollers = true
        scroll.drawsBackground = false
        scroll.borderType = .noBorder
        scroll.documentView = document
        scroll.wantsLayer = true
        scroll.layer?.cornerRadius = 16
        scroll.layer?.cornerCurve = .continuous
        scroll.layer?.masksToBounds = true
        scroll.layer?.borderWidth = 0.6
        scroll.layer?.borderColor = NSColor.white.withAlphaComponent(0.22).cgColor
        return scroll
    }
    private func addColumn(_ table: NSTableView, id: String, title: String, width: CGFloat) { let c = NSTableColumn(identifier: NSUserInterfaceItemIdentifier(id)); c.title = title; c.width = width; c.minWidth = min(width, 60); table.addTableColumn(c) }

    private func supported(_ url: URL) -> Bool { ["pdf", "png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"].contains(url.pathExtension.lowercased()) }
    private func safeFilename(_ value: String) -> String { value.replacingOccurrences(of: "/", with: "_").replacingOccurrences(of: ":", with: "_") }
    private func statusTitle(_ status: String) -> String { ["pending": "Ожидает", "running": "В работе", "paused": "Пауза", "error": "Ошибка", "complete": "Готово"][status] ?? status }
    private func duration(_ seconds: Int) -> String { seconds >= 3600 ? String(format: "%d:%02d", seconds / 3600, (seconds % 3600) / 60) : String(format: "%d:%02d", seconds / 60, seconds % 60) }
    private func append(_ text: String) { logView?.textStorage?.append(NSAttributedString(string: text)); logView?.scrollToEndOfDocument(nil) }
    private func alert(_ message: String) { let a = NSAlert(); a.messageText = "GLM OCR Local"; a.informativeText = message; a.alertStyle = .warning; a.runModal() }
    private func emptyPreviewHTML(_ text: String) -> String { "<html><head><meta name='color-scheme' content='light dark'><style>body{font:16px -apple-system;padding:56px;color:#888;text-align:center}</style></head><body>\(text)</body></html>" }
    private func markdownPreviewHTML(_ text: String) -> String { "<html><head><meta charset='utf-8'><meta name='color-scheme' content='light dark'><style>body{font:15px -apple-system;line-height:1.55;max-width:900px;margin:30px auto;padding:0 24px;white-space:pre-wrap}img{max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #888;padding:6px}</style></head><body>\(text.replacingOccurrences(of: "&", with: "&amp;").replacingOccurrences(of: "<", with: "&lt;").replacingOccurrences(of: ">", with: "&gt;"))</body></html>" }

    private static let timestamp: DateFormatter = { let f = DateFormatter(); f.dateFormat = "yyyyMMdd-HHmmss"; return f }()
}

@main
struct GLMOCRLocalMain {
    private static let appDelegate = AppDelegate()
    static func main() { let app = NSApplication.shared; app.delegate = appDelegate; app.setActivationPolicy(.regular); app.run() }
}
